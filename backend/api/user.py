"""User Management API endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class UserProfile(BaseModel):
    """User profile model."""
    user_id: str
    email: EmailStr
    first_name: str
    last_name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    phone: Optional[str] = None


class HSAAccount(BaseModel):
    """HSA account model."""
    account_id: str
    balance: float
    ytd_contributions: float
    ytd_spending: float
    contribution_limit: float


class Transaction(BaseModel):
    """Transaction model."""
    transaction_id: str
    date: str
    description: str
    amount: float
    category: str
    is_eligible: bool


@router.get("/profile", response_model=UserProfile)
async def get_user_profile(user_id: str):
    """Get user profile.
    
    Args:
        user_id: User identifier
        
    Returns:
        User profile
    """
    # TODO: Fetch from database
    return UserProfile(
        user_id=user_id,
        email="user@example.com",
        first_name="John",
        last_name="Doe",
        age=35,
        gender="male",
        phone="555-0123"
    )


@router.put("/profile")
async def update_user_profile(profile: UserProfile):
    """Update user profile.
    
    Args:
        profile: Updated profile data
        
    Returns:
        Success message
    """
    # TODO: Update in database
    return {"message": "Profile updated successfully"}


@router.get("/hsa-account", response_model=HSAAccount)
async def get_hsa_account(user_id: str):
    """Get HSA account information.
    
    Args:
        user_id: User identifier
        
    Returns:
        HSA account details
    """
    # TODO: Fetch from database
    return HSAAccount(
        account_id="HSA-123456",
        balance=2500.00,
        ytd_contributions=3000.00,
        ytd_spending=500.00,
        contribution_limit=4150.00
    )


@router.get("/transactions", response_model=List[Transaction])
async def get_transactions(user_id: str, limit: int = 10):
    """Get user transactions.
    
    Args:
        user_id: User identifier
        limit: Number of transactions to return
        
    Returns:
        List of transactions
    """
    # TODO: Fetch from database
    return [
        Transaction(
            transaction_id="TXN-001",
            date="2024-01-15",
            description="Prescription eyeglasses",
            amount=250.00,
            category="Vision",
            is_eligible=True
        ),
        Transaction(
            transaction_id="TXN-002",
            date="2024-01-10",
            description="Doctor visit",
            amount=150.00,
            category="Medical",
            is_eligible=True
        )
    ]


@router.get("/dashboard")
async def get_dashboard(user_id: str):
    """Get user dashboard data.
    
    Args:
        user_id: User identifier
        
    Returns:
        Dashboard data with all key metrics
    """
    return {
        "user_id": user_id,
        "hsa_balance": 2500.00,
        "ytd_spending": 500.00,
        "predicted_annual_spending": 4200.00,
        "recommended_contribution": 350.00,
        "recent_transactions": 5,
        "pending_reminders": 2,
        "savings_to_date": 924.00,  # Tax savings
        "utilization_score": 85,  # Out of 100
        "quick_stats": {
            "eligible_purchases": 12,
            "total_purchases": 15,
            "eligibility_rate": "80%"
        }
    }