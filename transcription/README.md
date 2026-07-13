```bash
docker build -t whisper-backend .
```
#### CPU Mode
```bash
docker run -p 2345:2345 whisper-backend
```
#### GPU Mode (nvidia-ctk required)
```bash
docker run --gpus all -p 2345:2345 whisper-backend
```
`http://localhost:2345/transcribe` 
add multipart/form-data "file" with a common video file as the value
