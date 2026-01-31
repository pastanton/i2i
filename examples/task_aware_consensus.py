#!/usr/bin/env python3
"""
Task-Aware Consensus Example

Demonstrates how i2i automatically detects task type and provides
calibrated confidence based on evaluation findings:

- Factual questions: Use consensus (HIGH = 97-100% accuracy)
- Verification: Use consensus (+6% hallucination detection)  
- Math/Reasoning: DON'T use consensus (-35% degradation!)
- Creative: DON'T use consensus (flattens diversity)

Run with: python examples/task_aware_consensus.py
"""

import asyncio
from i2i import (
    AICP,
    recommend_consensus,
    is_consensus_appropriate,
    get_confidence_calibration,
)


def demo_task_classification():
    """Show how different questions get classified."""
    print("=" * 60)
    print("Task Classification Demo")
    print("=" * 60)
    
    questions = [
        # Factual - USE consensus
        ("What is the capital of France?", "factual"),
        ("Who wrote Romeo and Juliet?", "factual"),
        
        # Verification - USE consensus  
        ("Is it true that Einstein failed math?", "verification"),
        ("True or false: The Great Wall is visible from space", "verification"),
        
        # Math/Reasoning - DON'T use consensus
        ("Calculate 15 * 7 + 23", "reasoning"),
        ("If John has 5 apples and gives 2 to Mary, how many left?", "reasoning"),
        ("Solve for x: 2x + 5 = 15", "reasoning"),
        
        # Creative - DON'T use consensus
        ("Write me a poem about the ocean", "creative"),
        ("Come up with 5 startup ideas", "creative"),
    ]
    
    for question, expected in questions:
        rec = recommend_consensus(question)
        status = "✓" if rec.should_use_consensus else "✗"
        print(f"\n{status} {question[:50]}...")
        print(f"   Category: {rec.task_category.value} (expected: {expected})")
        print(f"   Use consensus: {rec.should_use_consensus}")
        if not rec.should_use_consensus:
            print(f"   Warning: {rec.reason[:60]}...")


def demo_confidence_calibration():
    """Show how consensus level maps to calibrated confidence."""
    print("\n" + "=" * 60)
    print("Confidence Calibration")
    print("=" * 60)
    print("\nBased on evaluation of 400 questions across 5 benchmarks:")
    print()
    
    levels = [
        ("HIGH (≥85% agreement)", "high", "Trust the answer"),
        ("MEDIUM (60-84%)", "medium", "Probably correct"),
        ("LOW (30-59%)", "low", "Use with caution"),
        ("NONE (<30%)", "none", "Likely hallucination"),
    ]
    
    for name, level, meaning in levels:
        conf = get_confidence_calibration(level)
        print(f"  {name:25} → {conf:.2f} confidence ({meaning})")


async def demo_consensus_query():
    """Show consensus_query with task-aware fields."""
    print("\n" + "=" * 60)
    print("Consensus Query with Task-Aware Fields")
    print("=" * 60)
    
    # Note: This requires API keys to actually run
    # We'll show the API usage pattern
    
    print("""
# Factual question (consensus appropriate)
result = await protocol.consensus_query("What is the capital of Japan?")
print(result.consensus_appropriate)   # True
print(result.task_category)           # "factual"
print(result.confidence_calibration)  # 0.95 (if HIGH consensus)

# Math question (consensus NOT appropriate)
result = await protocol.consensus_query("Calculate 25 * 4")
print(result.consensus_appropriate)   # False
print(result.task_category)           # "reasoning"
print(result.metadata['consensus_warning'])  
# "WARNING: Consensus DEGRADES math/reasoning by 35%..."

# Explicit task category override
result = await protocol.consensus_query(
    "Is this statement accurate?",
    task_category="verification"  # Skip auto-detection
)
""")


def demo_quick_check():
    """Show the quick is_consensus_appropriate() helper."""
    print("\n" + "=" * 60)
    print("Quick Check Helper")
    print("=" * 60)
    
    questions = [
        "What year did WWII end?",
        "Prove that the square root of 2 is irrational",
        "Write a haiku about programming",
    ]
    
    print("\nUsing is_consensus_appropriate() for quick checks:\n")
    for q in questions:
        appropriate = is_consensus_appropriate(q)
        status = "✓ Use consensus" if appropriate else "✗ Skip consensus"
        print(f"  {status}: {q[:45]}...")


def main():
    print("\n🔍 i2i Task-Aware Consensus Demo\n")
    
    demo_task_classification()
    demo_confidence_calibration()
    demo_quick_check()
    
    # Run async demo
    asyncio.run(demo_consensus_query())
    
    print("\n" + "=" * 60)
    print("Key Takeaway")
    print("=" * 60)
    print("""
i2i doesn't universally improve accuracy—it provides CALIBRATED CONFIDENCE.

✓ For factual/verification tasks: HIGH consensus = 95-100% accuracy
✗ For math/reasoning: Use single model with chain-of-thought instead

The value is knowing WHEN to trust the answer, not just getting an answer.
""")


if __name__ == "__main__":
    main()
