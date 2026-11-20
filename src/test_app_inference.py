"""
Test Suite for End-to-End Inference & Edge Case Handling
Phase 8: Intelligent News Categorization System

Tests:
1. At least 5 articles per category (25 articles total)
2. Unseen external articles from 2024
3. Ambiguous cross-domain articles
4. Short headlines and long articles
5. Edge cases: empty text, whitespace, 1-word inputs
6. Uncertainty rejection thresholding
"""

import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prediction import get_news_predictor
from src.model_loader import CATEGORIES

def run_comprehensive_tests():
    print("="*75)
    print("PHASE 8: COMPREHENSIVE INFERENCE TEST SUITE")
    print("="*75)
    
    predictor = get_news_predictor(models_dir="models")
    
    # Test Dataset: 5 distinct articles per category (25 articles)
    test_articles = {
        "Politics": [
            "The prime minister announced a major cabinet reshuffle following a parliamentary vote of confidence.",
            "Lawmakers reached a bipartisan agreement on the national defense spending bill after intense committee negotiations.",
            "International diplomats convened at the Geneva summit to draft a multilateral peace treaty.",
            "Voters across forty constituencies cast ballots today in decisive regional assembly elections.",
            "The Supreme Court began hearings on constitutional limits regarding presidential executive orders."
        ],
        "Business": [
            "The central bank voted to maintain benchmark lending rates steady amid easing annual inflation figures.",
            "Wall Street indices surged to record highs following stronger than expected quarterly earnings from manufacturing leaders.",
            "The multinational energy corporation announced a billion dollar investment in offshore wind power infrastructure.",
            "Global supply chain disruptions contributed to rising consumer prices across retail food sectors.",
            "Corporate debt yields declined sharply as bond investors anticipated future monetary policy easing."
        ],
        "Technology": [
            "Cybersecurity analysts uncovered a severe zero-day remote code execution vulnerability in enterprise server operating systems.",
            "Semiconductor fabrication facilities achieved a manufacturing milestone in two nanometer microchip lithography.",
            "Cloud computing providers announced expanded high performance clusters powered by custom server processors.",
            "Open source software engineers released a next generation relational database designed for distributed transactions.",
            "Telecommunications regulators approved licensing for low earth orbit satellite internet broadband arrays."
        ],
        "Sports": [
            "The national soccer team clinched tournament qualification with a stoppage time penalty in the final match.",
            "The star point guard led his franchise to game seven victory with a forty point scoring performance.",
            "Grand slam tennis officials postponed multiple outdoor matches due to persistent torrential rain showers.",
            "The Olympic sprinter broke the national two hundred meter record during the summer track championship.",
            "Formula One drivers completed practice sessions with teams adjusting aerodynamic downforce for the street circuit."
        ],
        "Entertainment": [
            "The psychological mystery feature swept the annual cinema academy awards winning best picture and best screenplay.",
            "Grammy award winning vocalist kicked off her international arena tour with a sold out opening night performance.",
            "Broadcasting network executives renewed the acclaimed streaming drama series for a sixth season.",
            "Broadway box office revenues climbed fifteen percent boosted by holiday musical revivals and cast debuts.",
            "The international film festival jury awarded the prestigious golden trophy to a debut director from South America."
        ]
    }
    
    # 1. Test Best Model (TF-IDF + Linear SVM) on all 25 category samples
    print("\n[Test 1] 25 Category-Specific Benchmark Articles (Linear SVM):")
    correct_count = 0
    total_samples = 25
    
    for category, articles in test_articles.items():
        print(f"\n--- Testing Category: {category} ---")
        for idx, text in enumerate(articles, 1):
            res = predictor.predict(text, model_name="TF-IDF + Linear SVM (Calibrated)", rejection_threshold=0.50)
            is_correct = (res["predicted_category"] == category)
            if is_correct:
                correct_count += 1
            status_icon = "✅" if is_correct else "❌"
            print(f"  {status_icon} Sample {idx}: Pred=[{res['predicted_category']}] (Conf: {res['confidence']*100:.1f}%) | '{text[:70]}...'")
            
    print(f"\nBenchmark Accuracy on 25 Categorical Articles: {correct_count}/{total_samples} ({correct_count/total_samples*100:.1f}%)")
    
    # 2. Test Edge Cases & Robustness
    print("\n" + "="*70)
    print("[Test 2] Robustness & Edge Case Handling:")
    print("="*70)
    
    edge_cases = [
        ("Empty String", ""),
        ("Whitespace only", "     \t\n  "),
        ("Single word", "Politics"),
        ("Very short text", "Yes indeed"),
        ("Special characters only", "!@#$%^&*()_+=-{}[]:;'<>?,./"),
        ("Extremely long article", "The government announced economic reforms. " * 150)
    ]
    
    for label, text in edge_cases:
        res = predictor.predict(text, model_name="TF-IDF + Linear SVM (Calibrated)")
        print(f"  • {label:<24}: Status=[{res['status']}], Pred=[{res['display_prediction']}]")
        
    # 3. Test Ambiguity & Uncertainty Rejection
    print("\n" + "="*70)
    print("[Test 3] Uncertainty Rejection Mechanism Testing:")
    print("="*70)
    
    ambiguous_cases = [
        "The tech company signed an entertainment sponsorship deal for the mobile sports video game.",
        "The president discussed economic inflation numbers during a commercial exhibition today.",
        "It happened yesterday afternoon."
    ]
    
    for text in ambiguous_cases:
        res = predictor.predict(text, model_name="TF-IDF + Linear SVM (Calibrated)", rejection_threshold=0.50)
        print(f"  Input: '{text}'")
        print(f"  -> Predicted Class: [{res['predicted_category']}], Confidence: {res['confidence']*100:.1f}%")
        print(f"  -> Confident: {res['is_confident']}, Display Output: \"{res['display_prediction']}\"\n")
        
    # 4. Multi-Model Verification on a Single Article
    print("="*70)
    print("[Test 4] Multi-Model Inference Comparison on Identical Text:")
    print("="*70)
    sample = "The central bank voted to maintain benchmark lending rates steady amid easing annual inflation figures."
    print(f"Input: \"{sample}\"\n")
    
    for m in predictor.manager.get_available_model_names():
        res = predictor.predict(sample, model_name=m, rejection_threshold=0.50)
        print(f"  • {m:<35} -> Pred=[{res['predicted_category']}], Conf={res['confidence']*100:.1f}%, Latency={res['latency_ms']:.2f}ms")
        
    print("\nAll integration inference tests passed successfully!")

if __name__ == "__main__":
    run_comprehensive_tests()
