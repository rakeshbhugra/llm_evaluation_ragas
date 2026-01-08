def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into chunks with overlap.

    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Number of overlapping characters between chunks

    Returns:
        list: List of text chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        # Get chunk from start to start + chunk_size
        end = start + chunk_size
        chunk = text[start:end]

        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk)

        # Move start position (chunk_size - overlap for next chunk)
        start += chunk_size - overlap

    return chunks


# Example usage
if __name__ == "__main__":
    # Sample text
    text = "This is a sample text. " * 100

    # Create chunks
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    print(f"Total chunks created: {len(chunks)}")
    print(f"\nFirst 3 chunks:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(chunk[:100])
