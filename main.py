#!/usr/bin/env python3
"""
DocSynthesis-V1 Main Entry Point
IndiaAI IDP Challenge Submission

This script provides a command-line interface for processing documents
through the complete DocSynthesis-V1 pipeline.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

from src.config.settings import Settings
from src.preprocessing.pipeline import PreprocessingPipeline
from src.layout.analyzer import LayoutAnalyzer
from src.ocr.engine import OCREngine
from src.translation.nmt import NMTTranslator
from src.extraction.extractor import InformationExtractor
from src.extraction.summarizer import DocumentSummarizer
from src.xai.explainer import XAIExplainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('docsynthesis.log')
    ]
)

logger = logging.getLogger(__name__)


class DocSynthesisV1:
    """
    Main DocSynthesis-V1 Pipeline
    
    Orchestrates the complete document processing workflow:
    1. Preprocessing (restoration, correction)
    2. Layout Analysis (structure detection)
    3. OCR (text recognition)
    4. Translation (multilingual support)
    5. Extraction (structured data)
    6. Summarization (document summary)
    7. Explanation (XAI/FAM)
    """
    
    def __init__(self, config_path: str = None):
        """Initialize DocSynthesis-V1 with configuration."""
        logger.info("Initializing DocSynthesis-V1...")
        
        # Load configuration
        self.settings = Settings(config_path)
        
        # Initialize components
        self.preprocessor = PreprocessingPipeline(self.settings)
        self.layout_analyzer = LayoutAnalyzer(self.settings)
        self.ocr_engine = OCREngine(self.settings)
        self.translator = NMTTranslator(self.settings)
        self.extractor = InformationExtractor(self.settings)
        self.summarizer = DocumentSummarizer(self.settings)
        self.explainer = XAIExplainer(self.settings)
        
        logger.info("DocSynthesis-V1 initialized successfully")
    
    def process(
        self,
        input_path: str,
        output_dir: str = "output",
        language: str = None,
        translate: bool = False,
        extract_fields: bool = True,
        generate_summary: bool = True,
        explain: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a document through the complete pipeline.
        
        Args:
            input_path: Path to input document (PDF, image)
            output_dir: Directory for output files
            language: Source language (auto-detect if None)
            translate: Whether to translate to English
            extract_fields: Whether to extract structured fields
            generate_summary: Whether to generate summary
            explain: Whether to generate explanations
            
        Returns:
            Dictionary containing all processing results
        """
        logger.info(f"Processing document: {input_path}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            "input_path": input_path,
            "status": "processing"
        }
        
        try:
            # Stage 1: Preprocessing
            logger.info("Stage 1/7: Preprocessing...")
            preprocessed = self.preprocessor.process(input_path)
            results["preprocessing"] = {
                "restored": preprocessed["restored"],
                "corrected": preprocessed["corrected"],
                "quality_score": preprocessed["quality_score"]
            }
            
            # Stage 2: Layout Analysis
            logger.info("Stage 2/7: Layout Analysis...")
            layout = self.layout_analyzer.analyze(preprocessed["image"])
            results["layout"] = {
                "elements": layout["elements"],
                "hierarchy": layout["hierarchy"],
                "reading_order": layout["reading_order"],
                "confidence": layout["confidence"]
            }
            
            # Stage 3: OCR
            logger.info("Stage 3/7: OCR Processing...")
            ocr_result = self.ocr_engine.recognize(
                preprocessed["image"],
                layout_info=layout
            )
            results["ocr"] = {
                "text": ocr_result["text"],
                "markdown": ocr_result["markdown"],
                "tables": ocr_result["tables"],
                "confidence": ocr_result["confidence"],
                "compression_ratio": ocr_result["compression_ratio"]
            }
            
            # Stage 4: Translation (if requested)
            if translate:
                logger.info("Stage 4/7: Translation...")
                translated = self.translator.translate(
                    text=ocr_result["text"],
                    source_lang=language or "auto",
                    target_lang="english"
                )
                results["translation"] = {
                    "text": translated["text"],
                    "source_language": translated["detected_language"],
                    "bleu_score": translated.get("quality_score"),
                    "confidence": translated["confidence"]
                }
                text_for_extraction = translated["text"]
            else:
                logger.info("Stage 4/7: Translation skipped")
                text_for_extraction = ocr_result["text"]
                results["translation"] = None
            
            # Stage 5: Extraction
            if extract_fields:
                logger.info("Stage 5/7: Information Extraction...")
                extracted = self.extractor.extract(
                    text=text_for_extraction,
                    layout=layout,
                    document_type="auto"
                )
                results["extraction"] = {
                    "fields": extracted["fields"],
                    "provenance": extracted["provenance"],
                    "confidence": extracted["confidence"],
                    "field_coverage": extracted["coverage"]
                }
            else:
                logger.info("Stage 5/7: Extraction skipped")
                results["extraction"] = None
            
            # Stage 6: Summarization
            if generate_summary:
                logger.info("Stage 6/7: Document Summarization...")
                summary = self.summarizer.summarize(
                    text=text_for_extraction,
                    layout=layout,
                    method="hybrid"
                )
                results["summary"] = {
                    "short": summary["short"],
                    "medium": summary["medium"],
                    "long": summary["long"],
                    "key_points": summary["key_points"],
                    "rouge_score": summary.get("quality_score")
                }
            else:
                logger.info("Stage 6/7: Summarization skipped")
                results["summary"] = None
            
            # Stage 7: Explainability
            if explain:
                logger.info("Stage 7/7: Generating Explanations...")
                explanations = self.explainer.explain(
                    results=results,
                    image=preprocessed["image"]
                )
                results["explanations"] = {
                    "visual_attention": explanations["attention_maps"],
                    "token_attribution": explanations["token_scores"],
                    "natural_language": explanations["nl_explanations"],
                    "fam_score": explanations["fam_score"],
                    "fam_details": explanations["fam_details"]
                }
            else:
                logger.info("Stage 7/7: Explanations skipped")
                results["explanations"] = None
            
            results["status"] = "completed"
            
            # Save results
            self._save_results(results, output_path)
            
            logger.info(f"Processing completed successfully. Results saved to: {output_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}", exc_info=True)
            results["status"] = "failed"
            results["error"] = str(e)
            return results
    
    def _save_results(self, results: Dict[str, Any], output_path: Path):
        """Save processing results to files."""
        # Save JSON results
        with open(output_path / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save text output
        if results.get("ocr"):
            with open(output_path / "text.txt", "w", encoding="utf-8") as f:
                f.write(results["ocr"]["text"])
        
        # Save markdown output
        if results.get("ocr") and results["ocr"].get("markdown"):
            with open(output_path / "document.md", "w", encoding="utf-8") as f:
                f.write(results["ocr"]["markdown"])
        
        # Save translated text
        if results.get("translation"):
            with open(output_path / "translated.txt", "w", encoding="utf-8") as f:
                f.write(results["translation"]["text"])
        
        # Save extracted fields
        if results.get("extraction"):
            with open(output_path / "extracted_fields.json", "w", encoding="utf-8") as f:
                json.dump(results["extraction"]["fields"], f, indent=2, ensure_ascii=False)
        
        # Save summary
        if results.get("summary"):
            with open(output_path / "summary.txt", "w", encoding="utf-8") as f:
                f.write(f"SHORT SUMMARY:\n{results['summary']['short']}\n\n")
                f.write(f"MEDIUM SUMMARY:\n{results['summary']['medium']}\n\n")
                f.write(f"DETAILED SUMMARY:\n{results['summary']['long']}\n\n")
                f.write(f"KEY POINTS:\n")
                for point in results['summary']['key_points']:
                    f.write(f"- {point}\n")
        
        logger.info(f"Results saved to {output_path}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="DocSynthesis-V1: Intelligent Document Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic processing
  python main.py --input document.pdf --output results/
  
  # With translation
  python main.py --input document.pdf --language hindi --translate
  
  # Full pipeline with explanations
  python main.py --input document.pdf --translate --extract-fields --explain
  
  # Batch processing
  python main.py --input-dir documents/ --output-dir results/
        """
    )
    
    # Input/Output arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input document path (PDF, JPG, PNG)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Input directory for batch processing"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output",
        help="Output directory (default: output/)"
    )
    
    # Processing options
    parser.add_argument(
        "--language", "-l",
        type=str,
        help="Source language (auto-detect if not specified)"
    )
    parser.add_argument(
        "--translate", "-t",
        action="store_true",
        help="Translate to English"
    )
    parser.add_argument(
        "--extract-fields",
        action="store_true",
        default=True,
        help="Extract structured fields (default: True)"
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip field extraction"
    )
    parser.add_argument(
        "--summarize", "-s",
        action="store_true",
        default=True,
        help="Generate document summary (default: True)"
    )
    parser.add_argument(
        "--no-summarize",
        action="store_true",
        help="Skip summarization"
    )
    parser.add_argument(
        "--explain", "-e",
        action="store_true",
        help="Generate XAI explanations"
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input
    if not args.input and not args.input_dir:
        parser.error("Either --input or --input-dir must be specified")
    
    # Initialize system
    try:
        system = DocSynthesisV1(config_path=args.config)
    except Exception as e:
        logger.error(f"Failed to initialize DocSynthesis-V1: {e}")
        sys.exit(1)
    
    # Process single document
    if args.input:
        result = system.process(
            input_path=args.input,
            output_dir=args.output,
            language=args.language,
            translate=args.translate,
            extract_fields=args.extract_fields and not args.no_extract,
            generate_summary=args.summarize and not args.no_summarize,
            explain=args.explain
        )
        
        if result["status"] == "completed":
            print(f"\n✅ Processing completed successfully!")
            print(f"📁 Results saved to: {args.output}")
            if result.get("ocr"):
                print(f"📄 Confidence: {result['ocr']['confidence']:.2%}")
            if result.get("explanations"):
                print(f"🔍 FAM Score: {result['explanations']['fam_score']:.3f}")
        else:
            print(f"\n❌ Processing failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    
    # Batch processing
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output)
        
        # Find all supported documents
        documents = []
        for ext in [".pdf", ".jpg", ".jpeg", ".png", ".tiff"]:
            documents.extend(input_dir.glob(f"*{ext}"))
            documents.extend(input_dir.glob(f"*{ext.upper()}"))
        
        print(f"Found {len(documents)} documents to process")
        
        success_count = 0
        for doc in documents:
            doc_output = output_dir / doc.stem
            print(f"\nProcessing: {doc.name}")
            
            result = system.process(
                input_path=str(doc),
                output_dir=str(doc_output),
                language=args.language,
                translate=args.translate,
                extract_fields=args.extract_fields and not args.no_extract,
                generate_summary=args.summarize and not args.no_summarize,
                explain=args.explain
            )
            
            if result["status"] == "completed":
                success_count += 1
                print(f"✅ Completed: {doc.name}")
            else:
                print(f"❌ Failed: {doc.name} - {result.get('error')}")
        
        print(f"\n📊 Batch processing complete: {success_count}/{len(documents)} successful")


if __name__ == "__main__":
    main()

