from fastapi import FastAPI, Depends, HTTPException, Request, Query, Form, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from passlib.context import CryptContext

from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    Float,
    ForeignKey,
    UniqueConstraint,
    create_engine,
    func,
    case,
    and_
)
from sqlalchemy.orm import relationship, sessionmaker, declarative_base, Session
from enum import Enum
import pandas as pd
import numpy as np
import os


class GameVote(Enum):
    HAVENTPLAYED = 1
    WONTTRY = 2
    WONTPLAYAGAIN = 3
    ALREADYTRIED = 4
    WANTTOTRY = 5
    WANTTOPLAYAGAIN = 6
    CANTPLAYENOUGH = 7


def vote_num_to_str(vote_num: int | None) -> str:
    if vote_num is GameVote.HAVENTPLAYED.value:
        return "Haven't played"
    if vote_num is GameVote.WONTTRY.value:
        return "Won't try"
    if vote_num is GameVote.WONTPLAYAGAIN.value:
        return "Won't play again"
    if vote_num is GameVote.ALREADYTRIED.value:
        return "Already tried"
    if vote_num is GameVote.WANTTOTRY.value:
        return "Want to try"
    if vote_num is GameVote.WANTTOPLAYAGAIN.value:
        return "Want to play again"
    if vote_num is GameVote.CANTPLAYENOUGH.value:
        return "Can't play enough"
    return "Haven't played"


df_use_cols = ["objectname", "originalname", "objectid", "avgweight",
               "rank", "minplayers", "maxplayers", "version_yearpublished", "version_nickname", "itemtype"]

pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")


def normalize_password(password) -> str:
    """
    Ensure password is a str, limited to 72 BYTES for bcrypt.
    Handles non-ASCII safely by truncating bytes, then decoding.
    """
    if password is None:
        return ""

    # Always go through bytes for correct 72-byte truncation
    if isinstance(password, str):
        pw_bytes = password.encode("utf-8")
    else:
        # In case something weird (e.g. already bytes)
        pw_bytes = bytes(password)

    pw_bytes = pw_bytes[:72]  # bcrypt hard limit in BYTES
    return pw_bytes.decode("utf-8", "ignore")


def hash_password(password: str) -> str:
    pw = normalize_password(password)
    return pwd_context.hash(pw)


def verify_password(password: str, hashed: str) -> bool:
    pw = normalize_password(password)
    return pwd_context.verify(pw, hashed)


# -------------------------
# Database setup
# -------------------------
RUNNING_ON_PYTHONANYWHERE = False
if os.path.exists("/home/vadsuhanc"):
    RUNNING_ON_PYTHONANYWHERE = True

DB_PATH = "./data/boardgames.db" if not RUNNING_ON_PYTHONANYWHERE else "/home/vadsuhanc/repos/bgg_stealer/data/boardgames.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

print(">>> DB_PATH =", DB_PATH)
print(">>> DATABASE_URL =", DATABASE_URL)

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# -------------------------
# Models
# -------------------------

class Game(Base):
    __tablename__ = "games"

    id = Column(Integer, primary_key=True, index=True)
    objectid = Column(Integer, index=True, nullable=False)
    originalname = Column(String, index=True, nullable=False)
    objectname = Column(String, index=True, nullable=False)
    avgweight = Column(Float, nullable=False)
    rank = Column(Integer, nullable=False)
    minplayers = Column(Integer, nullable=False)
    maxplayers = Column(Integer, nullable=False)
    owner = Column(String, nullable=False)
    version_yearpublished = Column(Integer)
    version_nickname = Column(String)
    itemtype = Column(String)

    votes = relationship("Vote", back_populates="game",
                         cascade="all, delete-orphan")
    __table_args__ = (
        UniqueConstraint("objectname",
                         "owner", "objectid", name="uix_object_owner"),
    )


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)

    votes = relationship("Vote", back_populates="user",
                         cascade="all, delete-orphan")


class Vote(Base):
    __tablename__ = "votes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    vote_num = Column(Integer, nullable=False)
    # vote_str = Column(String, nullable=False)

    user = relationship("User", back_populates="votes")
    game = relationship("Game", back_populates="votes")

    __table_args__ = (
        UniqueConstraint("user_id", "game_id", name="uix_user_game"),
    )


Base.metadata.create_all(bind=engine)


# -------------------------
# Pydantic schemas (for JSON API, optional but useful)
# -------------------------

class GameCreate(BaseModel):
    objectname: str
    originalname: str
    objectid: int
    avgweight: float
    rank: int
    minplayers: int
    maxplayers: int
    owner: str
    version_yearpublished: Optional[int]
    version_nickname: Optional[str]
    itemtype: str


class GameOut(BaseModel):
    # id: int
    objectname: str
    originalname: str
    objectid: int
    avgweight: float
    rank: int
    minplayers: int
    maxplayers: int
    owners: str
    version_yearpublished: Optional[int]
    version_nickname: Optional[str]
    itemtype: str
    username: Optional[str]
    vote_num: Optional[int]
    vote_str: Optional[str]

    # likes: int
    # dislikes: int

    class Config:
        orm_mode = True


class UserCreate(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    # password: str

    class Config:
        orm_mode = True


# -------------------------
# FastAPI app & templates
# -------------------------

app = FastAPI(title="Board Game Likes (HTML + API)")

templates = Jinja2Templates(directory="templates")

# Optional static directory (CSS, JS, images)
static_folder = "/static" if not RUNNING_ON_PYTHONANYWHERE else "/home/vadsuhanc/repos/bgg_stealer/static"
app.mount(static_folder, StaticFiles(directory="static"), name="static")


# -------------------------
# Dependencies
# -------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -------------------------
# Helper functions
# -------------------------

def get_current_user(request: Request, db: Session):
    user_id = request.cookies.get("user_id")
    if not user_id:
        return None
    return db.query(User).filter(User.id == int(user_id)).first()


def get_all_users(db: Session):
    return db.query(User).all()


def get_all_games(db: Session, user_id: int) -> List[GameOut]:
    """
    Return all games, plus the *given user's* username + vote for each game.
    One GameOut per game.
    """

    # FULL OUTER JOIN votes filtered for this user
    query = (
        db.query(
            Game.objectid.label("objectid"),
            Game.originalname.label("originalname"),
            func.max(Game.objectname).label("objectname"),
            func.max(Game.avgweight).label("avgweight"),
            func.max(Game.minplayers).label("minplayers"),
            func.max(Game.maxplayers).label("maxplayers"),
            func.max(Game.rank).label("rank"),
            func.max(Game.version_yearpublished).label(
                "version_yearpublished"),
            func.max(Game.version_nickname).label("version_nickname"),
            func.max(Game.itemtype).label("itemtype"),
            User.username.label("username"),   # may be None if no vote
            Vote.vote_num.label("vote_num"),   # may be None if no vote
            # func.array_agg(Game.owner).label("owners"),
            func.group_concat(Game.owner.distinct()).label("owners"),
        )
        .outerjoin(
            Vote,
            and_(
                Vote.game_id == Game.objectid,
                Vote.user_id == user_id,   # only this user's vote
            ),
        )
        .outerjoin(
            User,
            User.id == Vote.user_id,
        ).group_by(
            Game.objectid,
            Game.originalname,
            User.username,   # may be None if no vote
            Vote.vote_num,   # may be None if no vote
        )
        .order_by(
            (case((Vote.vote_num.is_(None), 0), else_=Vote.vote_num)).desc(),
            (Game.avgweight >= 2.0).desc(),        # heavy games first
            Game.maxplayers.desc(),                # more players first
            case(                                  # rank: 0 or NULL -> big value -> bottom
                (Game.rank == 0, 9999999),
                (Game.rank.is_(None), 9999999),
                else_=Game.rank,
            ).asc(),)
    )

    # print(query.statement)
    rows = query.all()

    result: List[GameOut] = []
    for objectid, originalname, objectname, avgweight, minplayers, maxplayers, rank, version_yearpublished, version_nickname, itemtype, username, vote_num, owners in rows:
        # print(f"{objectid}: {objectname} -> {owners}")
        # print(objectid, objectname, avgweight, minplayers, maxplayers, rank,
        #       version_yearpublished, version_nickname, itemtype, username, vote_num, owners)
        result.append(
            GameOut(
                originalname=originalname,
                objectname=objectname,
                objectid=objectid,
                avgweight=avgweight,
                rank=rank,
                minplayers=minplayers,
                maxplayers=maxplayers,
                owners=owners,
                version_yearpublished=version_yearpublished,
                version_nickname=version_nickname,
                itemtype=itemtype,
                username=username or "",              # None → empty string
                vote_num=vote_num or 1,
                vote_str=vote_num_to_str(vote_num) or "",
            )
        )

    return result

# Seed some demo data on first run (optional)


def seed_demo_data(db: Session):
    if not db.query(User).first():
        user = User(
            username="Miki",
            password=hash_password("Miki")    # default password
        )
        db.add(user)
        user = User(
            username="Vera",
            password=hash_password("Vera")    # default password
        )
        db.add(user)

    if not db.query(Game).first():
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, "data")

        owners = ["jatszohazprojekt", "Boardgamebudapest",
                  "jatszma_kavezo", "gemklub_corvin"]
        dfs = []
        for o in owners:
            csv = os.path.join(DATA_DIR, f"collection_{o}.csv")

            df = pd.read_csv(csv,
                             usecols=df_use_cols)
            df["owner"] = o
            dfs.append(df)

        df_concated = pd.concat(dfs).drop_duplicates(
            subset=["owner", "objectid"])

        df_concated["version_yearpublished"] = pd.to_numeric(
            df_concated["version_yearpublished"], errors="coerce")
        df_concated["avgweight"] = pd.to_numeric(
            df_concated["avgweight"], errors="coerce")

        df_concated = df_concated[df_concated["avgweight"].fillna(-1) > 0]

        for _, row in df_concated.iterrows():
            # Skip if game already exists (by name or BGG ID)
            existing = db.query(Game).filter(
                Game.objectname == row["objectname"]).first()
            if existing:
                continue

            yearval = row.get("version_yearpublished")

            game = Game(
                objectname=row["objectname"],
                originalname=row["originalname"],

                minplayers=row.get("minplayers"),
                maxplayers=row.get("maxplayers"),
                avgweight=row.get("avgweight"),
                rank=row.get("rank"),
                objectid=row.get("objectid"),
                owner=row.get("owner"),
                version_yearpublished=int(
                    yearval) if pd.notna(yearval) else None,
                version_nickname=row.get("version_nickname", None),
                itemtype=row.get("itemtype"),
            )

            db.add(game)

    db.commit()


# -------------------------
# HTML routes (Jinja)
# -------------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request,
          db: Session = Depends(get_db),
          owner: str = Query("all"),
          game_type: str = Query("all"),
          user: str = Query("me"),
          ):
    """
    Main page: list games + like/dislike buttons.
    """
    # Seed some data for convenience (only if DB is empty)
    seed_demo_data(db)

    current_user = get_current_user(request, db)

    if not current_user:
        return RedirectResponse(url="/login", status_code=303)

    # games = get_all_games_with_counts(db)
    user_id_to_check = current_user.id if user == "me" else int(user)
    games = get_all_games(db, user_id_to_check)

    # collect all unique owners from the concatenated string
    all_owners = sorted({
        o.strip()
        for g in games
        for o in (g.owners or "").split(",")
        if o.strip()
    })
    all_game_types = sorted({g.itemtype for g in games})

    # filter by owner first (if not "all")
    if owner != "all":
        games = [
            g for g in games
            if g.owners and owner in [o.strip() for o in g.owners.split(",")]
        ]

    if game_type != "all":
        games = [g for g in games if g.itemtype == game_type]

    all_users = get_all_users(db)

    return templates.TemplateResponse(
        "games.html",
        {
            "request": request,
            "current_user": current_user,
            "user": user_id_to_check,
            "games": games,
            "owners": all_owners,
            "users": all_users,
            "game_types": all_game_types,
            "current_owner": owner,
            "current_game_type": game_type,
        },
    )


@app.post("/games/{game_id}/vote")
def vote_game_html(
    game_id: int,
    request: Request,
    # vote_submitted: int = Form(...),          # "true" or "false"
    user_id: int = Form(...),       # hidden field from form
    vote_num: int = Form(...),       # hidden field from form
    db: Session = Depends(get_db),
    owner: str = Query("all"),
    game_type: str = Query("all"),
    user: str = Query("me"),
):
    """
    Handle like/dislike from HTML form and redirect back to homepage.
    """
    current_user = get_current_user(request, db)

    if not current_user:
        return RedirectResponse(url="/login", status_code=303)

    # games = get_all_games_with_counts(db)
    user_id_to_check = current_user.id if user == "me" else int(user)

    game = db.query(Game).filter(Game.objectid == game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    vote = db.query(Vote).filter(
        Vote.game_id == game_id,
        Vote.user_id == user_id,
    ).first()

    if vote:
        vote.vote_num = vote_num
    else:
        vote = Vote(user_id=user_id, game_id=game_id, vote_num=vote_num)
        db.add(vote)

    db.commit()

    # Redirect back to list
    return RedirectResponse(url=f"/?owner={owner}&game_type={game_type}&user_id={user_id_to_check}", status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": None}
    )


@app.post("/login")
def login_submit(
    request: Request,
    response: Response,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == username).first()

    if not user or not verify_password(password, user.password):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid username or password"},
            status_code=400
        )

    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie("user_id", str(user.id), httponly=True)
    return response


@app.get("/logout")
def logout(response: Response):
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("user_id")
    return response


# -------------------------
# Optional JSON API (reuse from previous code)
# -------------------------

@app.post("/api/users", response_model=UserOut)
def create_user(user_in: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == user_in.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    user = User(
        username=user_in.username,
        password=hash_password(user_in.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.get("/api/users", response_model=List[UserOut])
def list_users(db: Session = Depends(get_db)):
    return db.query(User).all()


@app.post("/api/games", response_model=GameOut)
def create_game(game_in: GameCreate, db: Session = Depends(get_db)):
    existing = db.query(Game).filter(
        Game.objectname == game_in.objectname).first()
    if existing:
        raise HTTPException(
            status_code=400, detail="Game with this name already exists")

    game = Game(objectname=game_in.objectname, avgweight=game_in.avgweight, maxplayers=game_in.maxplayers, minplayers=game_in.minplayers,
                objectid=game_in.objectid, owner=game_in.owner, rank=game_in.rank, version_yearpublished=game_in.version_yearpublished, version_nickname=game_in.version_nickname, itemtype=game_in.itemtype)
    db.add(game)
    db.commit()
    db.refresh(game)
    return get_all_games(db)


@app.get("/api/games", response_model=List[GameOut])
def list_games(db: Session = Depends(get_db)):
    return get_all_games(db)


# @app.get("/api/games/{game_id}", response_model=GameOut)
# def get_game_api(game_id: int, db: Session = Depends(get_db)):
#     game = db.query(Game).filter(Game.id == game_id).first()
#     if not game:
#         raise HTTPException(status_code=404, detail="Game not found")
#     return get_game_with_counts(game, db)
