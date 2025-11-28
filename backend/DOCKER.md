# Arthya Backend - Docker Deployment

## Using Docker Compose (Recommended)

The easiest way to run the entire stack:

```bash
# Start MongoDB + Backend
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop everything
docker-compose down

# Stop and remove volumes (data will be lost)
docker-compose down -v
```

## Using Docker Only

```bash
# Build the image
docker build -t arthya-backend .

# Run MongoDB first
docker run -d --name arthya-mongo -p 27017:27017 mongo:7

# Run the backend
docker run -d --name arthya-backend \
  -p 3000:3000 \
  -e MONGODB_URI=mongodb://arthya-mongo:27017/arthya \
  -e JWT_SECRET=your-secret-key \
  --link arthya-mongo \
  arthya-backend
```

## Environment Variables

Update `docker-compose.yml` with your production values:
- `JWT_SECRET` - Use a strong secret key
- `CLIENT_URL` - Your frontend URL
- `MONGODB_URI` - MongoDB connection (default is fine for local)

## Access

- Backend API: http://localhost:3000
- Health Check: http://localhost:3000/health
- MongoDB: localhost:27017

## Production Deployment

For production, consider:
1. Use environment variable files (.env)
2. Set up proper MongoDB authentication
3. Use secrets management
4. Configure proper logging
5. Set up monitoring
6. Use reverse proxy (nginx)
