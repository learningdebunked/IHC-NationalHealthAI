"""Main API router for IHC Platform."""

from fastapi import APIRouter

from api import eligibility, spending, assistant, auth, user

# Create main router
router = APIRouter()

# Include sub-routers
router.include_router(
    eligibility.router,
    prefix="/eligibility",
    tags=["Eligibility Classifier"]
)

router.include_router(
    spending.router,
    prefix="/spending",
    tags=["Spending Predictor"]
)

router.include_router(
    assistant.router,
    prefix="/assistant",
    tags=["Health Assistant"]
)

router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"]
)

router.include_router(
    user.router,
    prefix="/user",
    tags=["User Management"]
)