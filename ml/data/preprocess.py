#!/usr/bin/env python3
"""Data Preprocessing Pipeline for IHC Platform."""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPipeline:
    def __init__(self, data_dir: str = '.'):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        
        # Create directories
        for d in [self.raw_dir, self.processed_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Data directory: {self.data_dir.absolute()}")
    
    def generate_eligibility_data(self, n_samples: int = 10000):
        """Generate eligibility classification data."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating Eligibility Data ({n_samples:,} samples)")
        logger.info(f"{'='*60}")
        
        np.random.seed(42)
        
        # Define categories with items and eligibility
        categories = {
            'Prescriptions': {
                'items': ['prescription medication', 'insulin', 'prescription drug', 
                         'antibiotics', 'prescription inhaler'],
                'eligible': 1
            },
            'Medical Devices': {
                'items': ['blood pressure monitor', 'wheelchair', 'hearing aid', 
                         'glucose meter', 'crutches'],
                'eligible': 1
            },
            'Vision': {
                'items': ['prescription eyeglasses', 'contact lenses', 
                         'prescription sunglasses'],
                'eligible': 1
            },
            'Dental': {
                'items': ['dental treatment', 'orthodontics', 'dentures', 
                         'tooth extraction'],
                'eligible': 1
            },
            'Medical Supplies': {
                'items': ['bandages', 'diabetic supplies', 'test strips', 
                         'medical gauze'],
                'eligible': 1
            },
            'OTC': {
                'items': ['vitamins', 'supplements', 'pain reliever', 
                         'cold medicine'],
                'eligible': 0
            },
            'Cosmetic': {
                'items': ['cosmetic surgery', 'botox', 'teeth whitening', 
                         'hair transplant'],
                'eligible': 0
            },
            'Wellness': {
                'items': ['gym membership', 'massage therapy', 'spa treatment', 
                         'yoga class'],
                'eligible': 0
            },
        }
        
        # Generate samples
        data = []
        for _ in range(n_samples):
            cat_name = np.random.choice(list(categories.keys()))
            cat_data = categories[cat_name]
            item = np.random.choice(cat_data['items'])
            
            data.append({
                'item_description': item,
                'is_eligible': cat_data['eligible'],
                'category': cat_name
            })
        
        df = pd.DataFrame(data)
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        train_val, test = train_test_split(
            df, test_size=0.15, random_state=42, 
            stratify=df['is_eligible']
        )
        train, val = train_test_split(
            train_val, test_size=0.176, random_state=42,  # 0.15/0.85 ≈ 0.176
            stratify=train_val['is_eligible']
        )
        
        # Save datasets
        out_dir = self.processed_dir / 'eligibility'
        out_dir.mkdir(exist_ok=True)
        
        train.to_csv(out_dir / 'train.csv', index=False)
        val.to_csv(out_dir / 'val.csv', index=False)
        test.to_csv(out_dir / 'test.csv', index=False)
        
        logger.info(f"✓ Train set: {len(train):,} samples")
        logger.info(f"✓ Val set:   {len(val):,} samples")
        logger.info(f"✓ Test set:  {len(test):,} samples")
        logger.info(f"✓ Saved to:  {out_dir.absolute()}")
        
        # Class distribution
        eligible = train['is_eligible'].sum()
        total = len(train)
        logger.info(f"✓ Class balance: {eligible}/{total} eligible ({eligible/total*100:.1f}%)")
        
        return train, val, test
    
    def generate_spending_data(self, n_samples: int = 50000):
        """Generate spending prediction data."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating Spending Data ({n_samples:,} samples)")
        logger.info(f"{'='*60}")
        
        np.random.seed(42)
        
        # Generate features
        df = pd.DataFrame({
            'age': np.random.randint(18, 85, n_samples),
            'gender': np.random.choice([0, 1], n_samples),
            'family_size': np.random.randint(1, 7, n_samples),
            'chronic_conditions': np.random.poisson(1.5, n_samples),
            'bmi': np.clip(np.random.normal(27, 5, n_samples), 15, 50),
            'smoker': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'has_insurance': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
            'income': np.random.lognormal(10.8, 0.8, n_samples).astype(int),
        })
        
        # Generate realistic spending based on features
        base_spending = 2000
        age_factor = df['age'] * 50
        chronic_factor = df['chronic_conditions'] * 1500
        bmi_factor = np.maximum(df['bmi'] - 25, 0) * 100
        smoker_factor = df['smoker'] * 2000
        insurance_factor = (1 - df['has_insurance']) * 5000
        noise = np.random.normal(0, 1000, n_samples)
        
        spending = (
            base_spending + age_factor + chronic_factor + 
            bmi_factor + smoker_factor + insurance_factor + noise
        )
        
        df['annual_spending'] = np.maximum(spending, 0)
        
        # Remove outliers
        df = df[df['annual_spending'] < 100000]
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        train_val, test = train_test_split(df, test_size=0.15, random_state=42)
        train, val = train_test_split(train_val, test_size=0.176, random_state=42)
        
        # Save datasets
        out_dir = self.processed_dir / 'spending'
        out_dir.mkdir(exist_ok=True)
        
        train.to_csv(out_dir / 'train.csv', index=False)
        val.to_csv(out_dir / 'val.csv', index=False)
        test.to_csv(out_dir / 'test.csv', index=False)
        
        logger.info(f"✓ Train set: {len(train):,} samples")
        logger.info(f"✓ Val set:   {len(val):,} samples")
        logger.info(f"✓ Test set:  {len(test):,} samples")
        logger.info(f"✓ Saved to:  {out_dir.absolute()}")
        
        # Spending statistics
        mean_spending = train['annual_spending'].mean()
        median_spending = train['annual_spending'].median()
        logger.info(f"✓ Mean spending:   ${mean_spending:,.2f}")
        logger.info(f"✓ Median spending: ${median_spending:,.2f}")
        
        return train, val, test


def main():
    """Run data generation pipeline."""
    print("\n" + "="*60)
    print("  IHC Platform - Data Generation Pipeline")
    print("="*60 + "\n")
    
    pipeline = DataPipeline()
    
    # Generate both datasets
    pipeline.generate_eligibility_data(n_samples=10000)
    pipeline.generate_spending_data(n_samples=50000)
    
    print("\n" + "="*60)
    print("  ✓ Data Generation Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review data in: ml/data/processed/")
    print("  2. Train eligibility: cd ../training && python3 train_eligibility_classifier.py")
    print("  3. Train spending: python3 train_spending_model.py")
    print("")


if __name__ == '__main__':
    main()