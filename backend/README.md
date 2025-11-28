# Arthya Backend API

AI-powered autonomous financial coaching backend for gig workers and informal sector employees.

## Features

- 🔐 **User Authentication** - JWT-based secure authentication
- 💰 **Transaction Management** - Track income, expenses, and transfers
- 🎯 **Goals & Milestones** - Set and track financial goals
- 📊 **Analytics** - Comprehensive financial insights and trends
- 🤖 **AI Coaching** - Personalized financial advice and recommendations
- 🔔 **Smart Notifications** - Proactive alerts and guidance

## Tech Stack

- **Runtime**: Node.js
- **Framework**: Express.js
- **Language**: TypeScript
- **Database**: MongoDB with Mongoose
- **Authentication**: JWT (JSON Web Tokens)
- **Security**: Helmet, bcrypt

## Getting Started

### Prerequisites

- Node.js (v18 or higher)
- MongoDB (local or Atlas)
- npm or yarn

### Installation

1. Clone the repository and navigate to backend:
```bash
cd backend
```

2. Install dependencies:
```bash
npm install
```

3. Set up environment variables:
```bash
cp .env.example .env
```

Edit `.env` and configure:
- `MONGODB_URI` - Your MongoDB connection string
- `JWT_SECRET` - Secret key for JWT tokens
- `PORT` - Server port (default: 3000)

4. Start development server:
```bash
npm run dev
```

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user
- `GET /api/auth/me` - Get current user
- `PUT /api/auth/profile` - Update user profile

### Transactions
- `GET /api/transactions` - Get all transactions
- `POST /api/transactions` - Create transaction
- `GET /api/transactions/:id` - Get single transaction
- `PUT /api/transactions/:id` - Update transaction
- `DELETE /api/transactions/:id` - Delete transaction
- `GET /api/transactions/summary` - Get transaction summary

### Goals
- `GET /api/goals` - Get all goals
- `POST /api/goals` - Create goal
- `GET /api/goals/:id` - Get single goal
- `PUT /api/goals/:id` - Update goal
- `PUT /api/goals/:id/progress` - Update goal progress
- `DELETE /api/goals/:id` - Delete goal

### Analytics
- `GET /api/analytics/dashboard` - Get financial dashboard
- `GET /api/analytics/trends` - Get spending trends
- `GET /api/analytics/income` - Get income analysis

### AI Coaching
- `POST /api/coaching/advice` - Get personalized financial advice
- `GET /api/coaching/insights` - Get spending insights
- `POST /api/coaching/notify` - Create smart notification

## Project Structure

```
backend/
├── src/
│   ├── config/         # Configuration files
│   ├── controllers/    # Route controllers
│   ├── middleware/     # Custom middleware
│   ├── models/         # Database models
│   ├── routes/         # API routes
│   └── server.ts       # App entry point
├── dist/               # Compiled JavaScript
├── package.json
├── tsconfig.json
└── .env
```

## Development

```bash
# Run in development mode with auto-reload
npm run dev

# Build for production
npm run build

# Run production build
npm start

# Run linter
npm run lint
```

## Authentication

All protected routes require a JWT token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

## Response Format

All API responses follow this structure:

```json
{
  "success": true,
  "data": { ... },
  "message": "Optional message"
}
```

Error responses:

```json
{
  "success": false,
  "message": "Error description"
}
```

## License

MIT
