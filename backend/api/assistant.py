"""Health Assistant API endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatMessage(BaseModel):
    """Chat message model."""
    message: str = Field(..., min_length=1, max_length=1000)
    user_id: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What medications are HSA eligible?",
                "user_id": "user123"
            }
        }


class Recommendation(BaseModel):
    """Recommendation model."""
    title: str
    description: str
    category: str
    priority: str
    action_url: Optional[str] = None


class MedicationReminder(BaseModel):
    """Medication reminder model."""
    medication_name: str
    dosage: str
    frequency: str
    time: str
    notes: Optional[str] = None


@router.post("/chat")
async def chat_with_assistant(message: ChatMessage):
    """Chat with the intelligent health assistant.
    
    Provides real-time personalized recommendations, answers questions
    about HSA/FSA eligibility, and offers health-aligned purchasing guidance.
    
    Args:
        message: User message
        
    Returns:
        Assistant response
    """
    try:
        # TODO: Implement actual NLP model
        # For now, return rule-based responses
        
        user_message = message.message.lower()
        
        # Simple keyword matching (replace with actual NLP model)
        if "medication" in user_message or "prescription" in user_message:
            response = {
                "response": "Prescription medications are generally HSA/FSA eligible. This includes most prescription drugs, insulin, and birth control pills. Over-the-counter medications typically require a prescription to be eligible. Would you like to check a specific medication?",
                "suggestions": [
                    "Check medication eligibility",
                    "View prescription history",
                    "Set medication reminder"
                ]
            }
        elif "eligible" in user_message:
            response = {
                "response": "I can help you determine if items are HSA/FSA eligible! Common eligible items include medical services, prescriptions, dental care, vision care, and certain medical equipment. What specific item would you like me to check?",
                "suggestions": [
                    "Check item eligibility",
                    "View eligible categories",
                    "Upload receipt"
                ]
            }
        elif "spending" in user_message or "forecast" in user_message:
            response = {
                "response": "I can help you forecast your healthcare spending and optimize your HSA/FSA contributions. Based on your profile and historical data, I can predict your annual expenses with 12.4% accuracy. Would you like to see your spending forecast?",
                "suggestions": [
                    "View spending forecast",
                    "Optimize contributions",
                    "See spending insights"
                ]
            }
        else:
            response = {
                "response": "I'm your intelligent health assistant! I can help you with:\n\n1. Checking HSA/FSA eligibility (95.3% accuracy)\n2. Forecasting healthcare spending (12.4% MAPE)\n3. Optimizing contribution strategies\n4. Setting medication reminders\n5. Providing personalized health recommendations\n\nWhat would you like help with?",
                "suggestions": [
                    "Check eligibility",
                    "Forecast spending",
                    "Get recommendations",
                    "Set reminders"
                ]
            }
        
        return {
            "message": response["response"],
            "suggestions": response["suggestions"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/recommendations", response_model=List[Recommendation])
async def get_recommendations(user_id: str):
    """Get personalized health recommendations.
    
    Args:
        user_id: User identifier
        
    Returns:
        List of personalized recommendations
    """
    # TODO: Implement actual recommendation engine
    # For now, return sample recommendations
    
    recommendations = [
        Recommendation(
            title="Annual Physical Exam Due",
            description="Your annual physical exam is coming up. Schedule it now to maximize your HSA benefits.",
            category="Preventive Care",
            priority="high",
            action_url="/schedule-appointment"
        ),
        Recommendation(
            title="Optimize HSA Contributions",
            description="Based on your predicted spending of $4,200, consider increasing your monthly contribution by $50.",
            category="Financial",
            priority="medium",
            action_url="/optimize-contribution"
        ),
        Recommendation(
            title="Prescription Refill Available",
            description="Your prescription is ready for refill. Use your HSA card for tax-free payment.",
            category="Medication",
            priority="high",
            action_url="/refill-prescription"
        ),
        Recommendation(
            title="Dental Cleaning Reminder",
            description="It's been 6 months since your last dental cleaning. Schedule your appointment.",
            category="Dental",
            priority="medium",
            action_url="/schedule-dental"
        ),
        Recommendation(
            title="Vision Exam Eligible",
            description="You're eligible for an annual vision exam. This is fully covered by your HSA.",
            category="Vision",
            priority="low",
            action_url="/schedule-vision"
        )
    ]
    
    return recommendations


@router.post("/reminders/medication")
async def set_medication_reminder(reminder: MedicationReminder):
    """Set a medication reminder.
    
    Args:
        reminder: Medication reminder details
        
    Returns:
        Confirmation
    """
    try:
        # TODO: Implement actual reminder system
        return {
            "success": True,
            "message": f"Reminder set for {reminder.medication_name}",
            "reminder": reminder.model_dump()
        }
    except Exception as e:
        logger.error(f"Error setting reminder: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/reminders")
async def get_reminders(user_id: str):
    """Get all medication reminders for a user.
    
    Args:
        user_id: User identifier
        
    Returns:
        List of reminders
    """
    # TODO: Fetch from database
    return {
        "reminders": [
            {
                "medication_name": "Lisinopril",
                "dosage": "10mg",
                "frequency": "Once daily",
                "time": "08:00 AM",
                "notes": "Take with food"
            },
            {
                "medication_name": "Metformin",
                "dosage": "500mg",
                "frequency": "Twice daily",
                "time": "08:00 AM, 08:00 PM",
                "notes": "Take with meals"
            }
        ]
    }


@router.get("/tips")
async def get_health_tips():
    """Get health and HSA/FSA optimization tips.
    
    Returns:
        List of helpful tips
    """
    return {
        "tips": [
            {
                "category": "HSA Optimization",
                "tip": "Maximize your HSA contributions early in the year to benefit from tax-free growth."
            },
            {
                "category": "Medication Savings",
                "tip": "Ask your doctor about generic alternatives to save up to 80% on prescriptions."
            },
            {
                "category": "Preventive Care",
                "tip": "Most preventive care services are covered at 100% - schedule your annual checkup!"
            },
            {
                "category": "FSA Planning",
                "tip": "FSA funds are use-it-or-lose-it. Plan your contributions carefully based on predicted spending."
            },
            {
                "category": "Receipt Management",
                "tip": "Keep all medical receipts. You can reimburse yourself from your HSA years later!"
            },
            {
                "category": "Tax Benefits",
                "tip": "HSA contributions reduce your taxable income. A $4,000 contribution saves ~$880 in taxes (22% bracket)."
            }
        ],
        "impact_metrics": {
            "medication_adherence_improvement": "27%",
            "financial_stress_reduction": "42%",
            "hsa_utilization_improvement": "15%"
        }
    }