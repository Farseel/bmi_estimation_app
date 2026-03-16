"""
Gemini API Service Module

This module integrates with Google's Gemini API to generate personalized
health recommendations based on BMI value and category.
"""

import os
from google import generativeai as genai
from typing import Optional


class GeminiService:
    """Service for generating health recommendations via Gemini API."""
    
    _is_configured = False
    
    @classmethod
    def configure(cls):
        """
        Configure Gemini API with API key from environment variable.
        Call this once at server startup.
        
        Expects GEMINI_API_KEY environment variable to be set.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("⚠ Warning: GEMINI_API_KEY environment variable not set")
            print("  Health recommendations will not be available")
            cls._is_configured = False
            return
        
        genai.configure(api_key=api_key)
        cls._is_configured = True
        print("✓ Gemini API configured successfully")
    
    @classmethod
    def generate_recommendation(cls, bmi: float, category: str, weight_kg: float) -> Optional[str]:
        """
        Generate detailed health recommendation using Gemini API.
        
        Args:
            bmi: BMI value (e.g., 25.3)
            category: BMI category (e.g., "Overweight")
            weight_kg: Weight in kilograms (for personalized calorie calculations)
            
        Returns:
            str: Detailed health recommendation text
        """
        if not cls._is_configured:
            return cls._local_recommendation(bmi, category, weight_kg)
        
        try:
            # Craft detailed prompt for comprehensive health recommendations
            prompt = f"""Based on the following health profile, provide detailed, personalized recommendations:

HEALTH METRICS:
- BMI: {bmi:.2f}
- Category: {category}
- Weight: {weight_kg:.1f} kg

Please provide DETAILED recommendations in this EXACT format:

**Daily Protein & Carbs:**
- Protein: [X]g per day (based on {weight_kg:.1f}kg body weight)
- Carbs: [X]g per day
- Healthy Fats: [X]g per day
(Calculate based on: Protein = 1.2-2.0g per kg depending on activity level, Carbs = 3-5g per kg, Fats = 0.5-1.5g per kg for maintenance)

**Recommended Foods:**
- Proteins: [list 5-6 specific foods]
- Carbohydrates: [list 5-6 specific foods]
- Healthy Fats: [list 3-4 specific sources]
- Vegetables/Fruits: [list 5-6 specific options]

**Workout Frequency & Type:**
- Cardio: [frequency and duration]
- Strength Training: [frequency and exercises]
- Flexibility: [frequency and type]
- Rest Days: [number per week]

**Key Action Items:**
- [3-4 specific, actionable steps for this person]

Make recommendations specific to "{category}" BMI category. Be practical and motivating."""
            
            # Try Gemini models in order of preference
            # Using 1.5-flash as primary (most available across API tiers)
            gemini_models = [
                'gemini-1.5-flash',
                'gemini-2.0-flash',
            ]
            response_text = None
            last_error = None
            
            for model_name in gemini_models:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    response_text = response.text
                    break
                except Exception as model_err:
                    last_error = model_err
                    continue
            
            if response_text:
                return response_text
            
            print(f"All Gemini models failed. Last error: {last_error}")
            return cls._local_recommendation(bmi, category, weight_kg)
            
        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")
            return cls._local_recommendation(bmi, category, weight_kg)

    @staticmethod
    def _local_recommendation(bmi: float, category: str, weight_kg: float) -> str:
        """Return detailed local recommendation when Gemini API is unavailable."""
        
        # Calculate personalized macronutrient requirements
        if category == 'Underweight':
            protein_g = int(weight_kg * 1.8)  # Higher protein for muscle gain
            carbs_g = int(weight_kg * 4.5)    # Higher carbs for caloric surplus
            fats_g = int(weight_kg * 1.2)
            
            return f"""**Daily Nutrition Plan (Weight Gain Focus):**
• Protein: {protein_g}g/day (1.8g per kg - muscle building)
• Carbohydrates: {carbs_g}g/day (4.5g per kg - energy for growth)
• Healthy Fats: {fats_g}g/day (1.2g per kg)

**Recommended Foods:**
Proteins: Chicken breast, salmon, eggs, Greek yogurt, lentils, tofu
Carbs: Oats, brown rice, sweet potatoes, quinoa, whole wheat bread, banana
Healthy Fats: Avocado, nuts, olive oil, chia seeds, fatty fish
Vegetables: Broccoli, spinach, carrots, bell peppers, sweet potato

**Workout Plan:**
• Strength Training: 4 days/week (60 min) - Focus on progressive weight training
• Cardio: 2 days/week (20-30 min) - Light to moderate intensity
• Rest: 1 day complete rest
• Focus: Build muscle mass through compound exercises (squats, deadlifts, bench press)

**Key Actions:**
1. Eat in a caloric surplus (300-500 kcal above maintenance)
2. Track macronutrients consistently
3. Prioritize compound strength exercises
4. Get adequate sleep (8 hours) for muscle recovery"""
        
        elif category == 'Normal':
            protein_g = int(weight_kg * 1.6)
            carbs_g = int(weight_kg * 3.5)
            fats_g = int(weight_kg * 1.0)
            
            return f"""**Daily Nutrition Plan (Maintenance):**
• Protein: {protein_g}g/day (1.6g per kg - muscle maintenance)
• Carbohydrates: {carbs_g}g/day (3.5g per kg - sustained energy)
• Healthy Fats: {fats_g}g/day (1.0g per kg)

**Recommended Foods:**
Proteins: Turkey, lean beef, chicken, white fish, beans, cottage cheese
Carbs: Brown rice, pasta, oats, fruits, whole grain cereals, legumes
Healthy Fats: Nuts, olive oil, coconut oil, seeds, avocado
Vegetables: Leafy greens, broccoli, Brussels sprouts, cauliflower, zucchini

**Workout Plan:**
• Strength Training: 3-4 days/week (45-60 min)
• Cardio: 3 days/week (30 min) - Moderate intensity
• Flexibility: 2-3 days/week (20 min yoga/stretching)
• Rest: 1-2 days complete rest

**Key Actions:**
1. Maintain balanced macronutrient intake
2. Exercise regularly for cardiovascular health
3. Focus on consistency with diet and workouts
4. Include variety in exercises to prevent plateaus"""
        
        elif category == 'Overweight':
            protein_g = int(weight_kg * 2.0)  # Higher protein to preserve muscle during weight loss
            carbs_g = int(weight_kg * 2.5)    # Reduced carbs for caloric deficit
            fats_g = int(weight_kg * 0.8)
            
            return f"""**Daily Nutrition Plan (Weight Loss Focus):**
• Protein: {protein_g}g/day (2.0g per kg - preserve muscle mass)
• Carbohydrates: {carbs_g}g/day (2.5g per kg - controlled portion)
• Healthy Fats: {fats_g}g/day (0.8g per kg)

**Recommended Foods:**
Proteins: Lean chicken, turkey, fish, egg whites, Greek yogurt, protein powder
Carbs: Sweet potatoes, brown rice (measured), oatmeal, whole grains, legumes
Healthy Fats: Fish oil, almonds, walnuts, avocado (measured portions)
Vegetables: All kinds - spinach, kale, broccoli, bell peppers, cucumber (unlimited)

**Workout Plan:**
• Cardio: 5 days/week (30-45 min) - Brisk walking, running, cycling, swimming
• Strength Training: 3 days/week (45 min) - Preserve muscle during weight loss
• HIIT: 1-2 days/week (20 min) - High intensity interval training for calorie burn
• Rest: 1 day complete rest

**Key Actions:**
1. Create a 500 kcal daily deficit for 0.5kg weight loss per week
2. Drink 2-3L water daily for satiety and metabolism
3. Track food intake using an app (MyFitnessPal, etc.)
4. Get 7-9 hours sleep - lack of sleep hinders weight loss"""
        
        else:  # Obese
            protein_g = int(weight_kg * 2.0)
            carbs_g = int(weight_kg * 2.0)
            fats_g = int(weight_kg * 0.7)
            
            return f"""**Daily Nutrition Plan (Significant Weight Loss Focus):**
• Protein: {protein_g}g/day (2.0g per kg - muscle preservation is critical)
• Carbohydrates: {carbs_g}g/day (2.0g per kg - controlled reduction)
• Healthy Fats: {fats_g}g/day (0.7g per kg)

**Recommended Foods:**
Proteins: Chicken breast, lean fish, egg whites, low-fat Greek yogurt, tofu, tempeh
Carbs: Brown rice (controlled), sweet potato, oats (measured), whole wheat, beans
Healthy Fats: Olive oil (measured), fish oil, walnuts, almonds, seeds
Vegetables: All kinds encouraged - broccoli, kale, spinach, asparagus, zucchini, peppers

**Workout Plan (Start Gradually):**
• Week 1-2: Walking (30 min, 5 days/week) - Build cardio base
• Week 3-6: Add Swimming (2 days/week) - Low impact, full body
• Week 7+: Introduce Light Strength Training (2 days/week, 20-30 min)
• Flexibility: 3 days/week (yoga, stretching for mobility)
• Rest: Always ensure 1-2 complete rest days

**Key Actions:**
1. Consult healthcare provider before starting exercise program
2. Create 750 kcal daily deficit (combined diet + exercise) for 0.75kg/week loss
3. Focus on sustainable lifestyle changes, not quick fixes
4. Start with low-impact activities to protect joints
5. Monitor progress through measurements, not just weight scale"""
