#!/usr/bin/env python3
"""
Example: Basic Document Processing
Process a single document through the complete pipeline.
"""

from main import DocSynthesisV1
from pathlib import Path

def main():
    # Initialize the system
    print("Initializing DocSynthesis-V1...")
    processor = DocSynthesisV1()
    
    # Example document (replace with your document path)
    input_document = "examples/sample_certificate.pdf"
    output_directory = "examples/output"
    
    if not Path(input_document).exists():
        print(f"Sample document not found: {input_document}")
        print("Please place a sample document in the examples/ directory")
        return
    
    print(f"\nProcessing document: {input_document}")
    
    # Process the document
    result = processor.process(
        input_path=input_document,
        output_dir=output_directory,
        language="auto",  # Auto-detect language
        translate=True,   # Translate to English if needed
        extract_fields=True,
        generate_summary=True,
        explain=True
    )
    
    # Display results
    print("\n" + "="*80)
    print("PROCESSING RESULTS")
    print("="*80)
    
    print(f"\nStatus: {result['status']}")
    
    if result['status'] == 'completed':
        # OCR Results
        if result.get('ocr'):
            print(f"\n📄 OCR Results:")
            print(f"   Confidence: {result['ocr']['confidence']:.2%}")
            print(f"   Compression Ratio: {result['ocr']['compression_ratio']:.1f}x")
            print(f"   Text Length: {len(result['ocr']['text'])} characters")
            print(f"   Tables Detected: {len(result['ocr']['tables'])}")
        
        # Translation Results
        if result.get('translation'):
            print(f"\n🌐 Translation Results:")
            print(f"   Detected Language: {result['translation']['detected_language']}")
            print(f"   Confidence: {result['translation']['confidence']:.2%}")
        
        # Extraction Results
        if result.get('extraction'):
            print(f"\n🔍 Extracted Fields:")
            for field, value in list(result['extraction']['fields'].items())[:5]:
                print(f"   {field}: {value}")
            print(f"   Field Coverage: {result['extraction']['coverage']:.1%}")
        
        # Summary Results
        if result.get('summary'):
            print(f"\n📝 Summary (Short):")
            print(f"   {result['summary']['short'][:200]}...")
        
        # XAI Results
        if result.get('explanations'):
            print(f"\n💡 Explainability:")
            print(f"   FAM Score: {result['explanations']['fam_score']:.3f}")
            if result['explanations']['nl_explanations']:
                print(f"\n   Explanation:")
                print(f"   {result['explanations']['nl_explanations'][0]}")
        
        print(f"\n✅ Complete results saved to: {output_directory}")
        print(f"   - Text: {output_directory}/text.txt")
        print(f"   - Markdown: {output_directory}/document.md")
        print(f"   - Fields: {output_directory}/extracted_fields.json")
        print(f"   - Summary: {output_directory}/summary.txt")
        print(f"   - Full Results: {output_directory}/results.json")
    
    else:
        print(f"\n❌ Processing failed: {result.get('error')}")

if __name__ == "__main__":
    main()

