
import pymupdf  # PyMuPDF

def parse_pdf_pages(pdf_path, max_pages=10):
    """
    Parse the first N pages of a PDF and extract text.

    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to parse (default: 10)

    Returns:
        list: List of dictionaries containing page number and text content
    """
    # Open the PDF document
    doc = pymupdf.open(pdf_path)

    # Get total number of pages
    total_pages = len(doc)
    print(f"Total pages in PDF: {total_pages}")

    # Limit to first N pages or total pages (whichever is smaller)
    pages_to_parse = min(max_pages, total_pages)
    print(f"Parsing first {pages_to_parse} pages...\n")

    # Extract text from each page
    pages_data = []
    for page_num in range(pages_to_parse):
        # Get the page
        page = doc[page_num]

        # Try different extraction methods
        # Method 1: Default text extraction
        text = page.get_text("text")

        # Method 2: Try blocks if default doesn't work well
        if len(text.strip()) < 10:  # If barely any text, try blocks method
            text = page.get_text("blocks")
            # blocks returns list of tuples, extract text from them
            if isinstance(text, list):
                text = "\n".join([block[4] if len(block) > 4 else "" for block in text if len(block) > 4])

        pages_data.append({
            'page_number': page_num + 1,
            'text': text,
            'char_count': len(text)
        })

        print(f"Page {page_num + 1}: {len(text)} characters extracted")

    # Close the document
    doc.close()

    return pages_data


# Example usage
if __name__ == "__main__":
    pdf_path = "input_data/harrypotter.pdf"

    # Parse first 10 pages
    pages = parse_pdf_pages(pdf_path, max_pages=10)

    # Display first 10 lines of each extracted page
    print("\n" + "="*70)
    print("First 10 lines of each page:")
    print("="*70)

    for page_data in pages:
        page_num = page_data['page_number']
        text = page_data['text']

        print(f"\n--- Page {page_num} ---")

        if text.strip():  # Only show if page has content
            lines = text.split('\n')
            # Show first 10 lines (or fewer if page has less)
            for i, line in enumerate(lines[:10], 1):
                if line.strip():  # Only show non-empty lines
                    print(f"{i}: {line}")
        else:
            print("(Empty page)")

        print()