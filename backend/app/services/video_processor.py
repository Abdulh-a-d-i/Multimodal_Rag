from pydub import AudioSegment
from pydub.silence import split_on_silence

def process_video(file_path: str, filename: str, file_id: str):
    try:
        # Extract audio
        video = VideoFileClip(file_path)
        audio = video.audio
        
        with NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            audio_path = temp_audio.name
            audio.write_audiofile(audio_path, codec='pcm_s16le')
        
        # Load audio file
        audio_segment = AudioSegment.from_wav(audio_path)
        
        # Split audio on silence
        chunks = split_on_silence(
            audio_segment,
            min_silence_len=1000,
            silence_thresh=-40,
            keep_silence=500
        )
        
        recognizer = sr.Recognizer()
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_path = f"{audio_path}_chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")
            
            with sr.AudioFile(chunk_path) as source:
                audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data)
                    
                    if text:
                        # Calculate timestamp
                        start_time = sum(c.duration_seconds for c in chunks[:i])
                        end_time = start_time + chunk.duration_seconds
                        timestamp = f"{int(start_time // 60):02d}:{int(start_time % 60):02d}-{int(end_time // 60):02d}:{int(end_time % 60):02d}"
                        
                        doc_id = f"{file_id}_chunk_{i}"
                        documents.append(text)
                        metadatas.append({
                            "source": filename,
                            "timestamp": timestamp,
                            "type": "video",
                            "file_id": file_id
                        })
                        ids.append(doc_id)
                    
                except sr.UnknownValueError:
                    continue
                finally:
                    os.unlink(chunk_path)
        
        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        os.unlink(audio_path)
        video.close()
        return True
    
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return False