# Arthya Backend API

AI-powered autonomous financial coaching backend for gig workers and informal sector employees.

## Features

- 🔐 **User Authentication** - JWT-based secure authentication
- 💰 **Transaction Management** - Track income, expenses, and transfers
- 🎯 **Goals & Milestones** - Set and track financial goals
- 📊 **Analytics** - Comprehensive financial insights and trends
- 🤖 **AI Coaching** - Personalized financial advice powered by Google Gemini
- 🔔 **Smart Notifications** - Proactive alerts and guidance
- 💵 **Budgets** - Category-wise budget tracking
- 🏆 **Achievements** - Gamification with badges and milestones
- 📈 **Predictions** - Income/expense forecasting and financial health score
- 📱 **SMS Import** - Bulk import transactions from bank SMS
- 💬 **AI Chat** - Conversational financial coach using Gemini AI

## Tech Stack

- **Runtime**: Node.js
- **Framework**: Express.js
- **Language**: TypeScript
- **Database**: PostgreSQL with Sequelize
- **Authentication**: JWT (JSON Web Tokens)
- **AI**: Google Gemini 1.5 Flash
- **Security**: Helmet, bcrypt

## Getting Started

### Prerequisites

- Node.js (v18 or higher)
- PostgreSQL
- npm or yarn
- Google Gemini API Key (get from [AI Studio](https://aistudio.google.com/apikey))

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
- `DB_HOST` - PostgreSQL host
- `DB_PORT` - PostgreSQL port (default: 5432)
- `DB_NAME` - Database name
- `DB_USER` - Database user
- `DB_PASSWORD` - Database password
- `JWT_SECRET` - Secret key for JWT tokens
- `PORT` - Server port (default: 3000)
- `GEMINI_API_KEY` - Google Gemini API key for AI features

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
- `POST /api/transactions/bulk-import` - Bulk import transactions (from SMS)
- `GET /api/transactions/by-category` - Get spending by category
- `GET /api/transactions/monthly-trends` - Get monthly trends

### Goals
- `GET /api/goals` - Get all goals
- `POST /api/goals` - Create goal
- `GET /api/goals/:id` - Get single goal
- `PUT /api/goals/:id` - Update goal
- `PUT /api/goals/:id/progress` - Update goal progress
- `DELETE /api/goals/:id` - Delete goal

### Budgets
- `GET /api/budgets` - Get all budgets
- `POST /api/budgets` - Create budget
- `GET /api/budgets/:id` - Get single budget
- `PUT /api/budgets/:id` - Update budget
- `DELETE /api/budgets/:id` - Delete budget
- `GET /api/budgets/summary` - Get budget summary

### Achievements
- `GET /api/achievements` - Get all achievements
- `POST /api/achievements/check` - Check and unlock achievements
- `GET /api/achievements/progress` - Get achievement progress
- `GET /api/achievements/leaderboard` - Get leaderboard

### Predictions & AI
- `GET /api/predictions/income` - Get income prediction
- `GET /api/predictions/expenses` - Get expense prediction
- `GET /api/predictions/cashflow` - Get cash flow projection
- `GET /api/predictions/nudges` - Get personalized nudges
- `GET /api/predictions/health-score` - Get financial health score

### Analytics
- `GET /api/analytics/dashboard` - Get financial dashboard
- `GET /api/analytics/trends` - Get spending trends
- `GET /api/analytics/income` - Get income analysis

### AI Coaching
- `POST /api/coaching/advice` - Get personalized financial advice
- `GET /api/coaching/insights` - Get spending insights
- `POST /api/coaching/notify` - Create smart notification
- `GET /api/coaching/ai-insights` - Get AI-powered insights
- `GET /api/coaching/profile` - Get complete financial profile
- `GET /api/coaching/weekly-summary` - Get weekly financial summary
- `GET /api/coaching/action-plan` - Get prioritized action plan
- `POST /api/coaching/chat` - Chat with AI coach

### Gemini AI Endpoints
- `GET /api/coaching/gemini-advice` - Get Gemini-powered personalized advice
- `GET /api/coaching/gemini-weekly` - Get AI weekly summary
- `GET /api/coaching/spending-analysis` - Get AI spending pattern analysis
- `GET /api/coaching/goal-advice/:goalId` - Get AI advice for specific goal

## Project Structure

```
backend/
├── src/
│   ├── config/         # Configuration files
│   │   └── database.ts # PostgreSQL/Sequelize config
│   ├── controllers/    # Route controllers
│   │   ├── authController.ts
│   │   ├── transactionController.ts
│   │   ├── goalController.ts
│   │   ├── budgetController.ts
│   │   ├── achievementController.ts
│   │   ├── predictionController.ts
│   │   ├── analyticsController.ts
│   │   └── coachingController.ts
│   ├── middleware/     # Custom middleware
│   │   ├── auth.ts     # JWT authentication
│   │   └── error.ts    # Error handling
│   ├── models/         # Sequelize models
│   │   ├── User.ts
│   │   ├── Transaction.ts
│   │   ├── Goal.ts
│   │   ├── Budget.ts
│   │   ├── Achievement.ts
│   │   ├── Card.ts
│   │   └── Notification.ts
│   ├── services/       # Business logic
│   │   ├── incomePrediction.ts
│   │   ├── categorization.ts
│   │   ├── nudgeService.ts
│   │   ├── coachingAI.ts     # Rule-based AI coaching
│   │   └── geminiService.ts  # Gemini AI integration
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
