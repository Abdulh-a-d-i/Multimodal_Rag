import pdfplumber

def process_pdf(file_path: str, filename: str, file_id: str):
    try:
        documents = []
        metadatas = []
        ids = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract text
                text = page.extract_text()
                
                # Extract tables
                tables = page.extract_tables()
                table_texts = []
                
                for table in tables:
                    for row in table:
                        table_texts.append(" | ".join(str(cell) for cell in row))
                
                full_text = text
                if table_texts:
                    full_text += "\n\nTables:\n" + "\n".join(table_texts)
                
                if full_text.strip():
                    doc_id = f"{file_id}_page_{page_num + 1}"
                    documents.append(full_text)
                    metadatas.append({
                        "source": filename,
                        "page": page_num + 1,
                        "type": "pdf",
                        "file_id": file_id
                    })
                    ids.append(doc_id)
        
        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        return True
    
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return False