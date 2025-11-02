#!/usr/bin/env python3
"""
Generate key concerns for bills using DSPy.

This script identifies the most critical issues in a bill based on
severe/high impact provisions and generates structured concerns.
"""

import json
import sys
from pathlib import Path
from typing import List
import dspy
from dotenv import load_dotenv
import os
import re
from shared import TOPICS, Topic, Severity

# Load environment variables
load_dotenv()


class KeyConcernGenerator(dspy.Signature):
    """Generate a key concern for a specific topic area impacted by a provision.

    IMPORTANT: Focus on the CURRENT PROVISION and how it impacts the SPECIFIC TOPIC AREA you're analyzing.
    Adjacent provisions are provided as context to understand how provisions relate and affect each other.

    Consider adjacent provisions when the current provision's impact depends on or is modified by what
    adjacent provisions establish or omit. Focus on direct functional relationships between provisions.

    Evaluation framework:
    - RULE OF LAW: Legal certainty, non-arbitrariness, equality before the law, judicial independence
    - FUNDAMENTAL JUSTICE: Presumption of innocence, right to fair trial, no punishment without law
    - SEPARATION OF POWERS: Checks and balances, independent oversight, no concentration of incompatible roles
    - PROPORTIONALITY: Penalties proportionate to harm, necessity, least restrictive means
    - DUE PROCESS: Notice, hearing, appeal rights, independent review before coercive action
    - DEMOCRATIC ACCOUNTABILITY: Transparency, parliamentary oversight, limits on executive power

    A key concern should:
    - Have a clear, punchy title (4-6 words maximum, shorter is better)
    - Explain what the provision does and why it's problematic for this topic area (2-3 sentences)
    - Focus on practical impact on rights, freedoms, or businesses
    - Be written in accessible language for non-experts
    - Use markdown formatting: **bold** for key terms, quoted provisions, or critical issues
    - Be specific to this particular provision's issue in this topic area
    - Consider the broader bill context when assessing severity
    - Quote specific language from the raw text when relevant
    - CITE PROVISIONS: When referencing adjacent provisions, cite them using markdown links in the format [index](#id)
      For example: [8](#section-20b-inserted). Only cite when the current provision's impact functionally depends on them.
    """

    bill_context: str = dspy.InputField(desc="Executive summary providing context about what the bill does")
    topic: Topic = dspy.InputField(desc="The SPECIFIC topic area you are analyzing for this concern. Focus your title and description on this topic's unique angle.")
    preceding_provisions: str = dspy.InputField(desc="JSON array of 2 preceding provisions for context: [{index, id, title, rawText}, ...]. CONTEXT ONLY - use to understand references and relationships.")
    current_provision: str = dspy.InputField(desc="JSON object for the provision being analyzed: {index, id, title, rawText}")
    following_provision: str = dspy.InputField(desc="JSON object for the following provision for context: {index, id, title, rawText}. CONTEXT ONLY - use to understand how provisions connect forward.")
    existing_concerns: str = dspy.InputField(desc="JSON array of concerns already generated for this provision from other topics: [{title, topic}, ...]. Your title MUST be clearly different from these existing titles.")
    impact_reasoning: str = dspy.InputField(desc="Detailed impact analysis from the impact assessment step. This covers all topic areas - extract the parts relevant to your specific topic.")
    impact_level: str = dspy.InputField(desc="The impact level for this specific topic (severe-negative or high-negative)")

    title: str = dspy.OutputField(desc="Concern title (4-6 words maximum, punchy and direct). MUST clearly indicate which topic-specific problem this addresses. Avoid repeating language from existing concern titles for this provision.")
    description: str = dspy.OutputField(desc="Concern description (2-3 sentences explaining what the provision does and why it's problematic for THIS SPECIFIC TOPIC AREA, use markdown formatting)")
    severity: Severity = dspy.OutputField(desc="Severity level: critical, high, medium or low")


def setup_dspy():
    """Initialize DSPy with Claude Sonnet model."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in environment variables")
        print("Please create a .env file with your Anthropic API key")
        print("See .env.example for template")
        sys.exit(1)

    # Using Claude Sonnet 4.5 for key concerns (better at identifying critical issues)
    # temperature=0 for consistent outputs
    lm = dspy.LM(model="claude-sonnet-4-5", api_key=api_key, temperature=0)
    dspy.configure(lm=lm)

    return lm


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'\s+', '-', text)
    text = re.sub(r'-+', '-', text)
    return text.strip('-')


def generate_key_concern(bill_context: str, topic: str, provision: dict, preceding: list, following: dict = None, existing_concerns: list = []) -> dict:
    """Generate a key concern for a specific topic area impact.

    Args:
        bill_context: Executive summary for context
        topic: Topic name
        provision: Provision dict with index, id, title, raw_text, impact_reasoning, impact_level
        preceding: List of preceding provision dicts (max 2)
        following: Following provision dict or None
        existing_concerns: List of already-generated concerns for this provision

    Returns:
        Concern dict with id, title, description, severity, topic, relatedProvisions
    """
    # Build current provision JSON
    current_provision_json = json.dumps({
        'index': provision['index'],
        'id': provision['id'],
        'title': provision['title'],
        'rawText': provision['raw_text']
    }, indent=2)

    # Build preceding provisions JSON
    preceding_json = json.dumps(preceding, indent=2)

    # Build following provision JSON
    following_json = json.dumps(following if following else {}, indent=2)

    # Build existing concerns JSON (just title and topic to avoid duplication)
    existing_json = json.dumps([
        {'title': c['title'], 'topic': c['topic']}
        for c in existing_concerns
    ], indent=2)

    # Generate concern
    generator = dspy.ChainOfThought(KeyConcernGenerator)
    result = generator(
        bill_context=bill_context,
        topic=topic,
        preceding_provisions=preceding_json,
        current_provision=current_provision_json,
        following_provision=following_json,
        existing_concerns=existing_json,
        impact_reasoning=provision['impact_reasoning'],
        impact_level=provision['impact_level']
    )

    return {
        "id": slugify(result.title),
        "title": result.title,
        "severity": result.severity.value,
        "description": result.description,
        "topic": topic,
        "relatedProvisions": [provision['id']]
    }


def process_bill(json_path: Path, dry_run: bool = False, force: bool = False):
    """Process a bill JSON file and generate key concerns.

    Args:
        json_path: Path to bill JSON file
        dry_run: If True, only show concerns without saving
        force: If True, regenerate even if exists
    """
    print(f"\nProcessing: {json_path.name}")
    print("=" * 80)

    # Load bill JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        bill_data = json.load(f)

    # Check if already has key concerns
    if 'keyConcerns' in bill_data and not force:
        print("✓ Key concerns already exist")
        print("  Skipping (use --force to regenerate)")
        return

    sections = bill_data.get('sections', [])
    if not sections:
        print("No sections found in bill")
        return

    # Use filename as bill title
    bill_title = json_path.stem

    print(f"Bill: {bill_title}")

    # Get executive summary for context
    executive_summary = bill_data.get('executiveSummary', '')
    if not executive_summary:
        print("Warning: No executive summary found. Key concerns will be generated without bill context.")
        executive_summary = "No executive summary available."

    print()

    # Collect ALL severe-negative or high-negative impacts per topic
    # Strategy: Generate one concern per topic per provision
    impactful_items = []

    for i, section in enumerate(sections):
        if section.get('category', {}).get('type') != 'provision':
            continue

        impact = section.get('impact', {})
        if not impact:
            continue

        impact_levels = impact.get('levels', {})
        confidence = impact.get('confidence', 0.5)

        # Find ALL severe-negative or high-negative impacts for this provision
        for topic in TOPICS:
            impact_level = impact_levels.get(topic, 'none')
            if impact_level in ['severe-negative', 'high-negative']:
                impactful_items.append({
                    'section_index': i,
                    'index': section.get('index', i + 1),
                    'id': section.get('id', ''),
                    'title': section.get('title', ''),
                    'raw_text': section.get('rawText', ''),
                    'impact_reasoning': impact.get('reasoning', ''),
                    'impact_level': impact_level,
                    'topic': topic,
                    'confidence': confidence
                })

    if not impactful_items:
        print("No provisions with severe-negative or high-negative impact found")
        print("Clearing any existing key concerns")
        bill_data['keyConcerns'] = []
        with open(json_path, 'w') as f:
            json.dump(bill_data, f, indent=2)
        print("✓ Cleared key concerns")
        return

    # Count unique provisions and impacts
    unique_provisions = len(set(item['id'] for item in impactful_items))
    severe_count = sum(1 for item in impactful_items if item['impact_level'] == 'severe-negative')
    high_count = sum(1 for item in impactful_items if item['impact_level'] == 'high-negative')

    print(f"Found {unique_provisions} provision(s) with high/severe impacts")
    print(f"  {severe_count} SEVERE-negative topic impact(s)")
    print(f"  {high_count} HIGH-negative topic impact(s)")
    print(f"Generating {len(impactful_items)} key concern(s) (one per topic per provision)")
    print()

    # Generate one concern per topic per provision
    key_concerns = []
    concerns_by_provision = {}  # Track concerns generated for each provision
    BATCH_SIZE = 10  # Save every 10 concerns

    for idx, item in enumerate(impactful_items, 1):
        print(f"[{idx}/{len(impactful_items)}] {item['topic']}: {item['title'][:50]}...")

        # Build sliding window context as structs
        section_index = item['section_index']
        preceding = []
        following = None

        # Get previous 2 provisions
        for j in range(section_index - 1, max(-1, section_index - 20), -1):
            if j >= 0 and j < len(sections):
                prev_section = sections[j]
                if prev_section.get('category', {}).get('type') == 'provision':
                    preceding.insert(0, {  # Insert at beginning to maintain order
                        'index': prev_section.get('index', j + 1),
                        'id': prev_section.get('id', ''),
                        'title': prev_section.get('title', ''),
                        'rawText': prev_section.get('rawText', '')
                    })
                    if len(preceding) == 2:
                        break

        # Get next 1 provision
        for j in range(section_index + 1, len(sections)):
            next_section = sections[j]
            if next_section.get('category', {}).get('type') == 'provision':
                following = {
                    'index': next_section.get('index', j + 1),
                    'id': next_section.get('id', ''),
                    'title': next_section.get('title', ''),
                    'rawText': next_section.get('rawText', '')
                }
                break

        # Get existing concerns for this provision
        provision_id = item['id']
        existing = concerns_by_provision.get(provision_id, [])

        concern = generate_key_concern(executive_summary, item['topic'], item, preceding, following, existing)
        key_concerns.append(concern)

        # Track this concern for future iterations
        if provision_id not in concerns_by_provision:
            concerns_by_provision[provision_id] = []
        concerns_by_provision[provision_id].append(concern)

        print(f"  ✓ {concern['severity'].upper()}: {concern['title']}")

        # Batch save progress (unless dry run)
        if not dry_run and idx % BATCH_SIZE == 0:
            # Sort current concerns by severity before saving
            severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            key_concerns.sort(key=lambda x: severity_order.get(x['severity'], 99))
            bill_data['keyConcerns'] = key_concerns
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(bill_data, f, indent=2, ensure_ascii=False)
            print(f"  💾 Saved progress ({idx}/{len(impactful_items)})")

        print()

    # Sort by severity (critical > high > medium > low)
    severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
    key_concerns.sort(key=lambda x: severity_order.get(x['severity'], 99))

    # Show results
    print()
    print("=" * 80)
    print(f"KEY CONCERNS ({len(key_concerns)})")
    print("=" * 80)
    print()

    for i, concern in enumerate(key_concerns, 1):
        print(f"{i}. [{concern['severity'].upper()}] {concern['title']}")
        print(f"   {concern['description']}")
        print(f"   Related provisions: {len(concern['relatedProvisions'])}")
        print()

    if dry_run:
        print("Dry run - not saving changes")
        return

    # Add to bill data
    bill_data['keyConcerns'] = key_concerns

    # Save
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(bill_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved {len(key_concerns)} key concerns to: {json_path.name}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python 8_generate_key_concerns.py <bill-json-path> [--dry-run] [--force]")
        print()
        print("Example:")
        print("  python 8_generate_key_concerns.py 'output/1. National Information Technology Authority (Amendment) Bill.json'")
        print("  python 8_generate_key_concerns.py output/bill.json --dry-run  # Test without saving")
        print("  python 8_generate_key_concerns.py output/bill.json --force    # Regenerate")
        sys.exit(1)

    # Check for flags
    dry_run = "--dry-run" in sys.argv
    force = "--force" in sys.argv or "-f" in sys.argv

    # Get single bill path
    bill_path = Path(sys.argv[1])

    if not bill_path.exists():
        print(f"Error: File not found: {bill_path}")
        sys.exit(1)

    # Setup DSPy
    print("Initializing DSPy...")
    setup_dspy()
    print("✓ DSPy initialized")
    print()

    # Process the bill
    try:
        process_bill(bill_path, dry_run=dry_run, force=force)
    except Exception as e:
        print(f"Error processing {bill_path.name}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
