import re

def chunk_lines(markdown_text, max_chunk_size = 1200, overlap_size = 200):
    url_match = re.search(r'\*\*Источник:\*\*\s*\[.*?\]\((.*?)\)', markdown_text)
    source_url = url_match.group(1) if url_match else None

    markdown_text = re.sub(r'!?\[(.*?)\]\(.*?\)', r'\1', markdown_text)
    lines = markdown_text.split('\n')

    raw_chunks = []
    current_headers = {}
    current_text_lines = []

    header_pattern = re.compile(r'^(#{1,6})\s+(.*)')
    in_code_block = False

    def save_raw_chunk():
        text = '\n'.join(current_text_lines).strip()
        if text:
            metadata = {f"H{lvl}": title for lvl, title in sorted(current_headers.items())}
            if source_url:
                metadata["url"] = source_url
            raw_chunks.append({"metadata": metadata, "text": text})
        current_text_lines.clear()

    for line in lines:
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            current_text_lines.append(line)
            continue

        match = header_pattern.match(line)

        if match and not in_code_block:
            save_raw_chunk()

            level = len(match.group(1))
            title = match.group(2).strip()

            current_headers[level] = title

            keys_to_remove = [k for k in current_headers.keys() if k > level]
            for k in keys_to_remove:
                del current_headers[k]

            current_text_lines.append(line)
        else:
            current_text_lines.append(line)

    save_raw_chunk()

    final_chunks = []

    for chunk in raw_chunks:
        text = chunk['text']
        metadata = chunk['metadata']

        if len(text) <= max_chunk_size:
            final_chunks.append(chunk)
            continue

        paragraphs = text.split('\n\n')
        current_part = []
        current_len = 0

        for p in paragraphs:
            p_len = len(p)

            if current_len + p_len > max_chunk_size and current_part:
                final_chunks.append({
                    "metadata": metadata.copy(),
                    "text": '\n\n'.join(current_part)
                })

                overlap_part = []
                overlap_len = 0
                for prev_p in reversed(current_part):
                    if overlap_len + len(prev_p) <= overlap_size:
                        overlap_part.insert(0, prev_p)
                        overlap_len += len(prev_p) + 2
                    else:
                        break

                current_part = overlap_part + [p]
                current_len = sum(len(x) for x in current_part) + (len(current_part) - 1) * 2
            else:
                current_part.append(p)
                current_len += p_len + (2 if current_part else 0)

        if current_part:
            final_chunks.append({
                "metadata": metadata.copy(),
                "text": '\n\n'.join(current_part)
            })

    return final_chunks

