#!/usr/bin/env python3
"""Evaluate trained eligibility classifier and generate real metrics."""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, accuracy_score, f1_score
)
from transformers import BertTokenizer
import json
import sys

sys.path.append(str(Path(__file__).parent.parent / 'training'))
from train_eligibility_classifier import EligibilityClassifier, EligibilityDataset

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


class EligibilityEvaluator:
    """Evaluate trained eligibility classifier."""
    
    def __init__(self, model_path, output_dir='./visualizations/real_metrics'):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*70}")
        print(f"  Eligibility Classifier - Real Metrics Evaluation")
        print(f"{'='*70}")
        print(f"Model: {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir.absolute()}\n")
    
    def load_model(self):
        """Load trained model."""
        print("Loading model...")
        
        if not self.model_path.exists():
            print(f"\n✗ Model not found: {self.model_path}")
            print("\nPlease train the model first:")
            print("  cd ../training")
            print("  python3 train_eligibility_classifier.py --config ../configs/eligibility_config.yaml\n")
            return False
        
        # Load model
        self.model = EligibilityClassifier()
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint.get('epoch', 'N/A')
                accuracy = checkpoint.get('best_accuracy', 'N/A')
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
                epoch = checkpoint.get('epoch', 'N/A')
                accuracy = checkpoint.get('accuracy', 'N/A')
            else:
                # Assume the dict itself is the state dict
                self.model.load_state_dict(checkpoint)
                epoch = 'N/A'
                accuracy = 'N/A'
        else:
            # Direct state dict
            self.model.load_state_dict(checkpoint)
            epoch = 'N/A'
            accuracy = 'N/A'
        
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        print(f"✓ Model loaded successfully")
        if epoch != 'N/A':
            print(f"  - Epoch: {epoch}")
        if accuracy != 'N/A':
            if isinstance(accuracy, (int, float)):
                print(f"  - Best Accuracy: {accuracy:.2f}%")
            else:
                print(f"  - Best Accuracy: {accuracy}")
        print()
        
        return True
    
    def load_test_data(self):
        """Load test dataset."""
        print("Loading test data...")
        
        test_path = Path('../data/processed/eligibility/test.csv')
        if not test_path.exists():
            print(f"✗ Test data not found: {test_path}\n")
            return False
        
        self.test_df = pd.read_csv(test_path)
        print(f"✓ Loaded {len(self.test_df):,} test samples\n")
        
        return True
    
    def evaluate(self):
        """Run evaluation on test set."""
        print("Evaluating model on test set...")
        
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for idx, row in self.test_df.iterrows():
                # Tokenize
                encoding = self.tokenizer(
                    row['item_description'],
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Predict
                outputs = self.model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1)
                
                all_preds.append(pred.item())
                all_probs.append(probs[0, 1].item())  # Probability of eligible class
                all_labels.append(row['is_eligible'])
                
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{len(self.test_df)} samples", end='\r')
        
        print(f"  Processed {len(self.test_df)}/{len(self.test_df)} samples")
        
        self.y_true = np.array(all_labels)
        self.y_pred = np.array(all_preds)
        self.y_probs = np.array(all_probs)
        
        # Calculate metrics
        self.accuracy = accuracy_score(self.y_true, self.y_pred)
        self.f1 = f1_score(self.y_true, self.y_pred)
        
        print(f"\n✓ Evaluation complete")
        print(f"  - Accuracy: {self.accuracy*100:.2f}%")
        print(f"  - F1 Score: {self.f1:.4f}\n")
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix."""
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Not Eligible', 'Eligible'],
                   yticklabels=['Not Eligible', 'Eligible'],
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - Real Model\nAccuracy: {self.accuracy*100:.2f}%',
                 fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        filename = self.output_dir / 'real_confusion_matrix.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename.name}")
        plt.close()
    
    def plot_roc_curve(self):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(self.y_true, self.y_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=3,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Real Model', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        filename = self.output_dir / 'real_roc_curve.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename.name}")
        plt.close()
    
    def plot_precision_recall_curve(self):
        """Plot precision-recall curve."""
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=3)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve - Real Model', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        filename = self.output_dir / 'real_precision_recall_curve.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename.name}")
        plt.close()
    
    def generate_classification_report(self):
        """Generate and save classification report."""
        report = classification_report(self.y_true, self.y_pred,
                                      target_names=['Not Eligible', 'Eligible'])
        
        report_text = f"""
{'='*70}
  REAL MODEL EVALUATION REPORT - Eligibility Classifier
{'='*70}

MODEL INFO:
{'-'*70}
Model Path: {self.model_path}
Test Samples: {len(self.y_true):,}

PERFORMANCE METRICS:
{'-'*70}
Accuracy: {self.accuracy*100:.2f}%
F1 Score: {self.f1:.4f}

CLASSIFICATION REPORT:
{'-'*70}
{report}

CONFUSION MATRIX:
{'-'*70}
{confusion_matrix(self.y_true, self.y_pred)}

{'='*70}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}
"""
        
        filename = self.output_dir / 'real_evaluation_report.txt'
        with open(filename, 'w') as f:
            f.write(report_text)
        print(f"✓ Saved: {filename.name}")
        print(report_text)
    
    def save_predictions(self):
        """Save predictions to CSV."""
        results_df = self.test_df.copy()
        results_df['predicted'] = self.y_pred
        results_df['probability'] = self.y_probs
        results_df['correct'] = (self.y_true == self.y_pred)
        
        filename = self.output_dir / 'predictions.csv'
        results_df.to_csv(filename, index=False)
        print(f"✓ Saved: {filename.name}")
    
    def run_all(self):
        """Run complete evaluation pipeline."""
        if not self.load_model():
            return
        
        if not self.load_test_data():
            return
        
        self.evaluate()
        
        print("Generating visualizations...\n")
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_precision_recall_curve()
        self.generate_classification_report()
        self.save_predictions()
        
        print(f"\n{'='*70}")
        print(f"  ✓ REAL METRICS EVALUATION COMPLETE!")
        print(f"{'='*70}")
        print(f"\nResults saved to: {self.output_dir.absolute()}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained eligibility classifier')
    parser.add_argument('--model', type=str,
                       default='../models/saved/eligibility_classifier_best.pth',
                       help='Path to trained model')
    parser.add_argument('--output', type=str,
                       default='./visualizations/real_metrics',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    evaluator = EligibilityEvaluator(args.model, args.output)
    evaluator.run_all()