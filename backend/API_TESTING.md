# Arthya Backend API - Testing Guide

## Base URL
```
http://localhost:3000
```

## 1. Authentication Endpoints

### Register User
```bash
POST /api/auth/register
Content-Type: application/json

{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "password123",
  "phone": "+1234567890",
  "occupation": "driver"
}
```

### Login
```bash
POST /api/auth/login
Content-Type: application/json

{
  "email": "john@example.com",
  "password": "password123"
}
```

Response includes token - save it for authenticated requests.

### Get Current User
```bash
GET /api/auth/me
Authorization: Bearer <your-token>
```

## 2. Transaction Endpoints

### Create Transaction
```bash
POST /api/transactions
Authorization: Bearer <your-token>
Content-Type: application/json

{
  "type": "income",
  "category": "Ride Fare",
  "amount": 150.50,
  "description": "Weekend rides",
  "date": "2024-11-29"
}
```

### Get All Transactions
```bash
GET /api/transactions?type=income&limit=20&page=1
Authorization: Bearer <your-token>
```

### Get Transaction Summary
```bash
GET /api/transactions/summary?startDate=2024-11-01&endDate=2024-11-30
Authorization: Bearer <your-token>
```

## 3. Goals Endpoints

### Create Goal
```bash
POST /api/goals
Authorization: Bearer <your-token>
Content-Type: application/json

{
  "title": "New Car",
  "description": "Save for a reliable car",
  "targetAmount": 25000,
  "currentAmount": 15000,
  "icon": "car-sport",
  "category": "purchase",
  "deadline": "2025-12-31"
}
```

### Update Goal Progress
```bash
PUT /api/goals/:id/progress
Authorization: Bearer <your-token>
Content-Type: application/json

{
  "amount": 500
}
```

## 4. Analytics Endpoints

### Get Dashboard
```bash
GET /api/analytics/dashboard
Authorization: Bearer <your-token>
```

### Get Spending Trends
```bash
GET /api/analytics/trends?period=30
Authorization: Bearer <your-token>
```

## 5. AI Coaching Endpoints

### Get Financial Advice
```bash
POST /api/coaching/advice
Authorization: Bearer <your-token>
```

### Get Spending Insights
```bash
GET /api/coaching/insights?period=30
Authorization: Bearer <your-token>
```

## Testing with cURL

Example with cURL:

```bash
# Register
curl -X POST http://localhost:3000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"name":"Test User","email":"test@example.com","password":"test123","occupation":"freelancer"}'

# Login and get token
TOKEN=$(curl -X POST http://localhost:3000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"test123"}' \
  | jq -r '.data.token')

# Use token for authenticated request
curl -X GET http://localhost:3000/api/analytics/dashboard \
  -H "Authorization: Bearer $TOKEN"
```

## Testing with Postman

1. Import the endpoints into Postman
2. Create an environment variable `token`
3. After login, save the token: `pm.environment.set("token", pm.response.json().data.token)`
4. Use `{{token}}` in Authorization header for protected routes
