docker build -t gitbotrag-backend .
docker tag gitbotrag-backend gcr.io/matchqr-2c510/gitbotrag-backend
docker push gcr.io/matchqr-2c510/gitbotrag-backend
gcloud run deploy gitbotrag-service --image gcr.io/matchqr-2c510/gitbotrag-backend --platform managed --region us-central1 --allow-unauthenticated --memory 4Gi
