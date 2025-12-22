#!/usr/bin/env python3
"""
Generate Demo Certificate for DocSynthesis-V1 Testing
Creates a sample government certificate for demonstration purposes
"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import random

def create_demo_certificate(output_path="examples/demo_certificate.png"):
    """Create a demo government certificate."""
    
    # Create image
    width, height = 2480, 3508  # A4 at 300 DPI
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Colors
    india_blue = (0, 56, 147)
    india_orange = (255, 103, 31)
    india_green = (19, 136, 8)
    black = (0, 0, 0)
    gray = (100, 100, 100)
    
    # Draw border
    border_margin = 100
    draw.rectangle(
        [(border_margin, border_margin), (width - border_margin, height - border_margin)],
        outline=india_blue,
        width=10
    )
    
    # Draw decorative header band
    draw.rectangle(
        [(border_margin, border_margin), (width - border_margin, border_margin + 200)],
        fill=india_blue
    )
    
    # Try to use a font, fallback to default if not available
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 80)
        heading_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 60)
        text_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 40)
        small_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 35)
    except:
        try:
            # Try alternative font paths
            title_font = ImageFont.truetype("/Library/Fonts/Arial Bold.ttf", 80)
            heading_font = ImageFont.truetype("/Library/Fonts/Arial Bold.ttf", 60)
            text_font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 40)
            small_font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 35)
        except:
            # Use default font as last resort
            title_font = heading_font = text_font = small_font = ImageFont.load_default()
            print("⚠️  Warning: Using default font. Install Arial for better results.")
    
    # Header text
    header_text = "GOVERNMENT OF INDIA"
    draw.text((width // 2, border_margin + 100), header_text, 
              fill='white', font=heading_font, anchor='mm')
    
    # Subheader
    subheader = "Ministry of Education"
    draw.text((width // 2, border_margin + 250), subheader, 
              fill=india_blue, font=text_font, anchor='mm')
    
    # Title
    title = "CERTIFICATE OF ACHIEVEMENT"
    draw.text((width // 2, 600), title, 
              fill=india_blue, font=title_font, anchor='mm')
    
    # Decorative line
    line_y = 700
    draw.line([(width // 2 - 400, line_y), (width // 2 + 400, line_y)], 
              fill=india_orange, width=5)
    
    # Main content
    content_start_y = 850
    line_height = 100
    
    texts = [
        ("This is to certify that", text_font, gray),
        ("", text_font, black),
        ("RAJESH KUMAR SHARMA", heading_font, india_green),
        ("", text_font, black),
        ("has successfully completed the", text_font, gray),
        ("Advanced Diploma in Artificial Intelligence", heading_font, india_blue),
        ("with a grade of 'A+' (92.5%)", text_font, india_orange),
    ]
    
    current_y = content_start_y
    for text, font, color in texts:
        if text:
            draw.text((width // 2, current_y), text, 
                     fill=color, font=font, anchor='mm')
        current_y += line_height
    
    # Details section
    details_y = current_y + 150
    left_margin = 300
    
    details = [
        ("Registration Number:", "AI/2024/7829-A"),
        ("Date of Issue:", "15th January 2024"),
        ("Valid Until:", "15th January 2029"),
    ]
    
    for label, value in details:
        draw.text((left_margin, details_y), label, 
                 fill=gray, font=small_font, anchor='lm')
        draw.text((left_margin + 600, details_y), value, 
                 fill=black, font=small_font, anchor='lm')
        details_y += 80
    
    # Issuer section
    issuer_y = height - 700
    
    draw.text((left_margin, issuer_y), "Issued by:", 
             fill=gray, font=small_font, anchor='lm')
    draw.text((left_margin, issuer_y + 60), "Dr. Amit Patel", 
             fill=india_blue, font=text_font, anchor='lm')
    draw.text((left_margin, issuer_y + 120), "Director, National Institute of Technology", 
             fill=black, font=small_font, anchor='lm')
    draw.text((left_margin, issuer_y + 180), "Authorized Signatory", 
             fill=gray, font=small_font, anchor='lm')
    
    # Seal placeholder
    seal_x = width - 500
    seal_y = issuer_y + 100
    draw.ellipse(
        [(seal_x - 100, seal_y - 100), (seal_x + 100, seal_y + 100)],
        outline=india_green,
        width=5
    )
    draw.text((seal_x, seal_y), "OFFICIAL\nSEAL", 
             fill=india_green, font=small_font, anchor='mm', align='center')
    
    # Footer
    footer_y = height - 250
    draw.line([(border_margin + 50, footer_y), (width - border_margin - 50, footer_y)], 
              fill=gray, width=2)
    
    draw.text((width // 2, footer_y + 50), "Document ID: CERT-2024-AI-7829", 
             fill=gray, font=small_font, anchor='mm')
    draw.text((width // 2, footer_y + 100), "This is a computer-generated certificate", 
             fill=gray, font=small_font, anchor='mm')
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Save the image
    img.save(output_path, 'PNG', dpi=(300, 300))
    print(f"✅ Demo certificate created: {output_path}")
    
    # Create a degraded version for preprocessing demo
    degraded_path = output_path.replace('.png', '_degraded.png')
    
    # Apply blur and reduce quality to simulate degradation
    degraded = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    
    # Save with reduced quality
    degraded.save(degraded_path, 'JPEG', quality=60)
    print(f"✅ Degraded certificate created: {degraded_path}")
    
    return output_path, degraded_path


if __name__ == "__main__":
    try:
        create_demo_certificate()
        print("\n🎉 Demo certificates generated successfully!")
        print("📁 Files created in 'examples/' directory")
        print("🚀 You can now use these files to test the Gradio interface.")
    except Exception as e:
        print(f"\n❌ Error generating certificates: {e}")
        print("💡 Make sure PIL/Pillow is installed: pip install Pillow")
