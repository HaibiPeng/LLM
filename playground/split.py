def split_text_into_chunks(file_path, chunk_size=1000):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        words = text.split()
        chunks = []
        current_chunk = []
        word_count = 0

        for word in words:
            current_chunk.append(word)
            word_count += 1

            if word_count >= chunk_size and word.endswith('.'):
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                word_count = 0

        # Handle the remaining words
        if current_chunk:
            # Try to find the last period in the remaining words
            last_period_index = -1
            for i in range(len(current_chunk) - 1, -1, -1):
                if current_chunk[i].endswith('.'):
                    last_period_index = i
                    break

            if last_period_index != -1:
                chunks.append(' '.join(current_chunk[:last_period_index + 1]))
                remaining = ' '.join(current_chunk[last_period_index + 1:])
                if remaining:
                    # If there are still words left, start a new chunk
                    current_chunk = remaining.split()
                    word_count = len(current_chunk)
            else:
                chunks.append(' '.join(current_chunk))

        # Save each chunk as a separate text file
        for i, chunk in enumerate(chunks):
            chunk_file_name = f'./input/chunk_{i + 1}.txt'
            with open(chunk_file_name, 'w', encoding='utf-8') as chunk_file:
                chunk_file.write(chunk)

        print(f'Successfully split the file into {len(chunks)} chunks.')
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

split_text_into_chunks('./tolkien.txt')