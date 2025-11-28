# Arthya API Documentation

Complete API reference for the Arthya financial coaching platform.

**Base URL:** `http://localhost:3000/api`

**Authentication:** Most endpoints require a JWT token in the Authorization header:
```
Authorization: Bearer <your-jwt-token>
```

---

## Table of Contents

1. [Authentication](#authentication)
2. [Transactions](#transactions)
3. [Goals](#goals)
4. [Budgets](#budgets)
5. [Achievements](#achievements)
6. [Predictions & AI](#predictions--ai)
7. [Analytics](#analytics)
8. [Coaching](#coaching)

---

## Authentication

### Register User

Creates a new user account.

```
POST /api/auth/register
```

**Authentication Required:** No

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| name | string | Yes | User's full name |
| email | string | Yes | Valid email address (unique) |
| password | string | Yes | Minimum 6 characters |
| phone | string | No | Phone number |
| occupation | string | No | One of: `driver`, `delivery`, `freelancer`, `domestic`, `vendor`, `daily_wage`, `other` |

**Example Request:**
```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "securepass123",
  "phone": "+919876543210",
  "occupation": "driver"
}
```

**Success Response (201):**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "name": "John Doe",
    "email": "john@example.com",
    "occupation": "driver",
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  }
}
```

**Error Response (400):**
```json
{
  "success": false,
  "message": "User already exists"
}
```

---

### Login User

Authenticates a user and returns a JWT token.

```
POST /api/auth/login
```

**Authentication Required:** No

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| email | string | Yes | Registered email address |
| password | string | Yes | User's password |

**Example Request:**
```json
{
  "email": "john@example.com",
  "password": "securepass123"
}
```

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "name": "John Doe",
    "email": "john@example.com",
    "occupation": "driver",
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  }
}
```

**Error Response (401):**
```json
{
  "success": false,
  "message": "Invalid credentials"
}
```

---

### Get Current User

Returns the authenticated user's profile.

```
GET /api/auth/me
```

**Authentication Required:** Yes

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "name": "John Doe",
    "email": "john@example.com",
    "phone": "+919876543210",
    "occupation": "driver",
    "monthlyIncome": 25000,
    "createdAt": "2025-11-28T10:00:00.000Z"
  }
}
```

---

### Update Profile

Updates the authenticated user's profile.

```
PUT /api/auth/profile
```

**Authentication Required:** Yes

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| name | string | No | Updated name |
| phone | string | No | Updated phone |
| occupation | string | No | Updated occupation |
| monthlyIncome | number | No | Estimated monthly income |

**Example Request:**
```json
{
  "name": "John D.",
  "monthlyIncome": 30000
}
```

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "name": "John D.",
    "email": "john@example.com",
    "monthlyIncome": 30000
  }
}
```

---

## Transactions

### Get All Transactions

Returns paginated list of user's transactions.

```
GET /api/transactions
```

**Authentication Required:** Yes

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| type | string | - | Filter by: `income`, `expense`, `transfer` |
| category | string | - | Filter by category name |
| startDate | string | - | ISO date string (YYYY-MM-DD) |
| endDate | string | - | ISO date string (YYYY-MM-DD) |
| limit | number | 50 | Results per page (max 100) |
| page | number | 1 | Page number |

**Example Request:**
```
GET /api/transactions?type=expense&category=Food&limit=10&page=1
```

**Success Response (200):**
```json
{
  "success": true,
  "count": 10,
  "total": 45,
  "page": 1,
  "pages": 5,
  "data": [
    {
      "id": 1,
      "userId": 1,
      "type": "expense",
      "category": "Food",
      "amount": "250.00",
      "description": "Swiggy order",
      "date": "2025-11-28T12:00:00.000Z",
      "recipient": null,
      "status": "completed",
      "source": "sms_import",
      "notes": null,
      "createdAt": "2025-11-28T12:00:00.000Z"
    }
  ]
}
```

---

### Get Single Transaction

Returns details of a specific transaction.

```
GET /api/transactions/:id
```

**Authentication Required:** Yes

**URL Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| id | string | Transaction ID |

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "userId": 1,
    "type": "expense",
    "category": "Food",
    "amount": "250.00",
    "description": "Swiggy order",
    "date": "2025-11-28T12:00:00.000Z",
    "status": "completed",
    "source": "manual"
  }
}
```

**Error Response (404):**
```json
{
  "success": false,
  "message": "Transaction not found"
}
```

---

### Create Transaction

Creates a new transaction.

```
POST /api/transactions
```

**Authentication Required:** Yes

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| type | string | Yes | `income`, `expense`, or `transfer` |
| amount | number | Yes | Transaction amount (positive) |
| category | string | Yes | Category name |
| description | string | No | Transaction description |
| date | string | No | ISO date (defaults to now) |
| recipient | string | No | Recipient name/identifier |
| status | string | No | `completed`, `pending`, `failed` (default: completed) |
| notes | string | No | Additional notes |

**Example Request:**
```json
{
  "type": "expense",
  "amount": 500,
  "category": "Transport",
  "description": "Uber ride to airport",
  "date": "2025-11-28"
}
```

**Success Response (201):**
```json
{
  "success": true,
  "data": {
    "id": 2,
    "userId": 1,
    "type": "expense",
    "amount": "500.00",
    "category": "Transport",
    "description": "Uber ride to airport",
    "date": "2025-11-28T00:00:00.000Z",
    "status": "completed",
    "source": "manual",
    "createdAt": "2025-11-28T12:00:00.000Z"
  }
}
```

---

### Update Transaction

Updates an existing transaction.

```
PUT /api/transactions/:id
```

**Authentication Required:** Yes

**URL Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| id | string | Transaction ID |

**Request Body:** Same as Create Transaction (all fields optional)

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "id": 2,
    "amount": "600.00",
    "category": "Transport",
    "description": "Uber ride to airport - updated"
  }
}
```

---

### Delete Transaction

Deletes a transaction.

```
DELETE /api/transactions/:id
```

**Authentication Required:** Yes

**Success Response (200):**
```json
{
  "success": true,
  "message": "Transaction deleted successfully"
}
```

---

### Get Transaction Summary

Returns income/expense totals for a period.

```
GET /api/transactions/summary
```

**Authentication Required:** Yes

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| startDate | string | ISO date string |
| endDate | string | ISO date string |

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "income": 50000,
    "expense": 12500,
    "transfer": 0,
    "incomeCount": 5,
    "expenseCount": 25,
    "transferCount": 0,
    "savings": 37500
  }
}
```

---

### Bulk Import Transactions

Imports multiple transactions at once (for SMS import).

```
POST /api/transactions/bulk-import
```

**Authentication Required:** Yes

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| transactions | array | Yes | Array of transaction objects |

**Transaction Object:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| type | string | Yes | `income` or `expense` |
| amount | number | Yes | Transaction amount |
| description | string | Yes | Transaction description |
| category | string | No | Category (auto-detected if not provided) |
| date | string | Yes | ISO date string |
| source | string | No | Source identifier (default: sms_import) |
| notes | string | No | Additional notes |

**Example Request:**
```json
{
  "transactions": [
    {
      "type": "expense",
      "amount": 250,
      "description": "Swiggy order",
      "date": "2025-11-28"
    },
    {
      "type": "income",
      "amount": 15000,
      "description": "Uber trip earnings",
      "date": "2025-11-27"
    }
  ]
}
```

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "imported": 2,
    "skipped": 0,
    "errors": [],
    "message": "Successfully imported 2 transactions, skipped 0 duplicates"
  }
}
```

**Notes:**
- Duplicates are detected by matching amount, description, and date (±1 minute)
- Categories are auto-detected using AI categorization if not provided
- Supports 50+ Indian merchant patterns (Swiggy, Zomato, Ola, etc.)

---

### Get Spending By Category

Returns expense breakdown by category.

```
GET /api/transactions/by-category
```

**Authentication Required:** Yes

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| startDate | string | ISO date string |
| endDate | string | ISO date string |

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "categories": [
      {
        "category": "Food",
        "total": 5000,
        "count": 20,
        "percentage": "40.0"
      },
      {
        "category": "Transport",
        "total": 3000,
        "count": 15,
        "percentage": "24.0"
      }
    ],
    "totalSpending": 12500
  }
}
```

---

### Get Monthly Trends

Returns income/expense trends by month.

```
GET /api/transactions/monthly-trends
```

**Authentication Required:** Yes

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| months | number | 6 | Number of months to include |

**Success Response (200):**
```json
{
  "success": true,
  "data": [
    {
      "month": "2025-11",
      "income": 50000,
      "expense": 12500,
      "savings": 37500,
      "savingsRate": 75
    },
    {
      "month": "2025-10",
      "income": 45000,
      "expense": 15000,
      "savings": 30000,
      "savingsRate": 67
    }
  ]
}
```

---

## Goals

### Get All Goals

Returns all user's financial goals.

```
GET /api/goals
```

**Authentication Required:** Yes

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| status | string | Filter by: `active`, `completed`, `cancelled` |

**Success Response (200):**
```json
{
  "success": true,
  "count": 2,
  "data": [
    {
      "id": 1,
      "userId": 1,
      "name": "Emergency Fund",
      "targetAmount": "50000.00",
      "currentAmount": "25000.00",
      "deadline": "2026-06-01T00:00:00.000Z",
      "category": "savings",
      "priority": "high",
      "status": "active",
      "progress": 50,
      "createdAt": "2025-11-01T00:00:00.000Z"
    }
  ]
}
```

---

### Get Single Goal

Returns details of a specific goal.

```
GET /api/goals/:id
```

**Authentication Required:** Yes

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "name": "Emergency Fund",
    "targetAmount": "50000.00",
    "currentAmount": "25000.00",
    "deadline": "2026-06-01T00:00:00.000Z",
    "category": "savings",
    "status": "active",
    "progress": 50
  }
}
```

---

### Create Goal

Creates a new financial goal.

```
POST /api/goals
```

**Authentication Required:** Yes

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| name | string | Yes | Goal name |
| targetAmount | number | Yes | Target amount to save |
| currentAmount | number | No | Current saved amount (default: 0) |
| deadline | string | No | Target date (ISO format) |
| category | string | No | Goal category |
| priority | string | No | `low`, `medium`, `high` |
| description | string | No | Goal description |

**Example Request:**
```json
{
  "name": "New Bike",
  "targetAmount": 80000,
  "deadline": "2026-03-01",
  "category": "purchase",
  "priority": "medium"
}
```

**Success Response (201):**
```json
{
  "success": true,
  "data": {
    "id": 2,
    "name": "New Bike",
    "targetAmount": "80000.00",
    "currentAmount": "0.00",
    "deadline": "2026-03-01T00:00:00.000Z",
    "status": "active",
    "progress": 0
  }
}
```

---

### Update Goal

Updates an existing goal.

```
PUT /api/goals/:id
```

**Authentication Required:** Yes

**Request Body:** Same as Create Goal (all fields optional)

---

### Update Goal Progress

Adds amount to a goal's current savings.

```
PUT /api/goals/:id/progress
```

**Authentication Required:** Yes

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| amount | number | Yes | Amount to add (can be negative) |

**Example Request:**
```json
{
  "amount": 5000
}
```

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "name": "Emergency Fund",
    "currentAmount": "30000.00",
    "targetAmount": "50000.00",
    "progress": 60
  }
}
```

---

### Delete Goal

Deletes a goal.

```
DELETE /api/goals/:id
```

**Authentication Required:** Yes

**Success Response (200):**
```json
{
  "success": true,
  "message": "Goal deleted successfully"
}
```

---

## Budgets

### Get All Budgets

Returns all user's budgets with spent amounts.

```
GET /api/budgets
```

**Authentication Required:** Yes

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| isActive | boolean | Filter by active status |

**Success Response (200):**
```json
{
  "success": true,
  "count": 2,
  "data": [
    {
      "id": 1,
      "userId": 1,
      "category": "Food",
      "amount": "5000.00",
      "period": "monthly",
      "startDate": "2025-11-01T00:00:00.000Z",
      "endDate": null,
      "isActive": true,
      "alertThreshold": 80,
      "spent": 2500,
      "remaining": 2500,
      "percentUsed": 50
    }
  ]
}
```

---

### Get Single Budget

Returns details of a specific budget.

```
GET /api/budgets/:id
```

**Authentication Required:** Yes

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "category": "Food",
    "amount": "5000.00",
    "spent": 2500,
    "remaining": 2500,
    "percentUsed": 50
  }
}
```

---

### Create Budget

Creates a new category budget.

```
POST /api/budgets
```

**Authentication Required:** Yes

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| category | string | Yes | Category to budget |
| amount | number | Yes | Budget limit |
| period | string | No | `weekly`, `monthly`, `yearly` (default: monthly) |
| startDate | string | No | Budget start date (default: now) |
| endDate | string | No | Budget end date |
| alertThreshold | number | No | Alert at X% usage (default: 80) |
| notes | string | No | Budget notes |

**Example Request:**
```json
{
  "category": "Entertainment",
  "amount": 3000,
  "period": "monthly",
  "alertThreshold": 75
}
```

**Success Response (201):**
```json
{
  "success": true,
  "data": {
    "id": 2,
    "category": "Entertainment",
    "amount": "3000.00",
    "period": "monthly",
    "isActive": true,
    "alertThreshold": 75
  }
}
```

**Error Response (400):**
```json
{
  "success": false,
  "message": "Active budget for this category already exists"
}
```

---

### Update Budget

Updates an existing budget.

```
PUT /api/budgets/:id
```

**Authentication Required:** Yes

**Request Body:** Same as Create Budget (all fields optional)

---

### Delete Budget

Deletes a budget.

```
DELETE /api/budgets/:id
```

**Authentication Required:** Yes

**Success Response (200):**
```json
{
  "success": true,
  "message": "Budget deleted successfully"
}
```

---

### Get Budget Summary

Returns overview of all active budgets.

```
GET /api/budgets/summary
```

**Authentication Required:** Yes

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "totalBudget": 15000,
    "totalSpent": 8500,
    "totalRemaining": 6500,
    "overallPercentUsed": 56.67,
    "budgetCount": 3,
    "categoryBreakdown": [
      {
        "category": "Food",
        "budgeted": 5000,
        "spent": 4500,
        "remaining": 500,
        "percentUsed": 90,
        "status": "warning"
      },
      {
        "category": "Transport",
        "budgeted": 7000,
        "spent": 3000,
        "remaining": 4000,
        "percentUsed": 42.86,
        "status": "good"
      },
      {
        "category": "Entertainment",
        "budgeted": 3000,
        "spent": 1000,
        "remaining": 2000,
        "percentUsed": 33.33,
        "status": "good"
      }
    ]
  }
}
```

**Status Values:**
- `good` - Under 80% of budget used
- `warning` - 80-100% of budget used
- `exceeded` - Over 100% of budget used

---

## Achievements

### Get All Achievements

Returns all achievements with unlock status.

```
GET /api/achievements
```

**Authentication Required:** Yes

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "achievements": [
      {
        "type": "first_transaction",
        "title": "First Step",
        "description": "Record your first transaction",
        "icon": "🎯",
        "category": "onboarding",
        "unlocked": true,
        "unlockedAt": "2025-11-28T12:00:00.000Z",
        "progress": 100
      },
      {
        "type": "saver_gold",
        "title": "Gold Saver",
        "description": "Save ₹50,000",
        "icon": "🥇",
        "category": "savings",
        "unlocked": false,
        "unlockedAt": null,
        "progress": 30
      }
    ],
    "totalUnlocked": 5,
    "totalAchievements": 13,
    "completionPercentage": 38.46
  }
}
```

**Achievement Types:**
| Type | Title | Description | Category |
|------|-------|-------------|----------|
| first_transaction | First Step | Record first transaction | onboarding |
| week_streak | Weekly Warrior | Log for 7 consecutive days | consistency |
| month_streak | Monthly Master | Log for 30 consecutive days | consistency |
| first_goal | Goal Setter | Create first savings goal | goals |
| goal_achieved | Goal Crusher | Complete first goal | goals |
| budget_created | Budget Boss | Create first budget | budgeting |
| under_budget | Budget Keeper | Stay under budget for a month | budgeting |
| saver_bronze | Bronze Saver | Save ₹1,000 | savings |
| saver_silver | Silver Saver | Save ₹10,000 | savings |
| saver_gold | Gold Saver | Save ₹50,000 | savings |
| transaction_100 | Century Club | Log 100 transactions | activity |
| transaction_500 | Power Tracker | Log 500 transactions | activity |
| category_explorer | Category Explorer | Use 10 expense categories | exploration |

---

### Check Achievements

Checks and unlocks any earned achievements.

```
POST /api/achievements/check
```

**Authentication Required:** Yes

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "newlyUnlocked": [
      {
        "id": 1,
        "type": "first_transaction",
        "title": "First Step",
        "description": "Record your first transaction",
        "icon": "🎯",
        "unlockedAt": "2025-11-28T12:00:00.000Z"
      }
    ],
    "message": "Congratulations! You unlocked 1 new achievement(s)!"
  }
}
```

**When No New Achievements:**
```json
{
  "success": true,
  "data": {
    "newlyUnlocked": [],
    "message": "No new achievements unlocked"
  }
}
```

---

### Get Achievement Progress

Returns progress towards various achievements.

```
GET /api/achievements/progress
```

**Authentication Required:** Yes

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "transactions": {
      "current": 45,
      "next_milestone": 100
    },
    "savings": {
      "current": 35000,
      "next_milestone": 50000
    },
    "categories": {
      "current": 7,
      "target": 10
    },
    "goals": {
      "created": 3,
      "completed": 1
    }
  }
}
```

---

### Get Leaderboard

Returns achievement leaderboard.

```
GET /api/achievements/leaderboard
```

**Authentication Required:** Yes

**Success Response (200):**
```json
{
  "success": true,
  "data": [
    {
      "userId": 5,
      "achievementCount": 10
    },
    {
      "userId": 1,
      "achievementCount": 7
    }
  ]
}
```

---

## Predictions & AI

### Get Income Prediction

Predicts future income based on historical data.

```
GET /api/predictions/income
```

**Authentication Required:** Yes

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| months | number | 3 | Months ahead to predict |

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "predictedAmount": 48000,
    "confidence": 75,
    "trend": "increasing",
    "seasonalPattern": "weekend_heavy",
    "volatility": 15.5,
    "recommendations": [
      "Your income has been growing steadily. Consider increasing your savings rate.",
      "Weekend earnings are 40% higher - plan expenses accordingly."
    ]
  }
}
```

**Trend Values:** `increasing`, `decreasing`, `stable`

**Seasonal Patterns:** `weekend_heavy`, `weekday_heavy`, `month_start`, `month_end`, `consistent`

---

### Get Expense Prediction

Predicts future expenses by category.

```
GET /api/predictions/expenses
```

**Authentication Required:** Yes

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| months | number | 3 | Months ahead to predict |

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "predictions": [
      {
        "month": "2025-12",
        "predicted": 15000
      },
      {
        "month": "2026-01",
        "predicted": 14500
      }
    ],
    "categoryPredictions": [
      {
        "category": "Food",
        "avgMonthly": 5000,
        "trend": "stable"
      },
      {
        "category": "Transport",
        "avgMonthly": 3500,
        "trend": "decreasing"
      }
    ],
    "historicalAverage": 14750,
    "dataPointsUsed": 6
  }
}
```

---

### Get Cash Flow Projection

Projects future cash flow (income - expenses).

```
GET /api/predictions/cashflow
```

**Authentication Required:** Yes

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| months | number | 3 | Months ahead to project |

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "currentAverages": {
      "income": 50000,
      "expense": 15000,
      "netMonthly": 35000
    },
    "projections": [
      {
        "month": "2025-12",
        "projectedIncome": 52000,
        "projectedExpense": 16000,
        "netCashFlow": 36000,
        "runningBalance": 71000
      },
      {
        "month": "2026-01",
        "projectedIncome": 48000,
        "projectedExpense": 15500,
        "netCashFlow": 32500,
        "runningBalance": 103500
      }
    ],
    "savingsRate": 70
  }
}
```

---

### Get Personalized Nudges

Returns AI-generated financial nudges and tips.

```
GET /api/predictions/nudges
```

**Authentication Required:** Yes

**Success Response (200):**
```json
{
  "success": true,
  "data": [
    {
      "id": "budget_warning_food",
      "type": "warning",
      "category": "budget",
      "title": "Food Budget Alert",
      "message": "You've used 85% of your Food budget. ₹750 remaining for 5 days.",
      "actionText": "View Budget",
      "actionType": "navigate",
      "priority": "high"
    },
    {
      "id": "goal_progress_emergency",
      "type": "celebration",
      "category": "goals",
      "title": "Goal Progress! 🎉",
      "message": "You're 75% towards your Emergency Fund goal!",
      "actionText": "View Goal",
      "actionType": "navigate",
      "priority": "medium"
    },
    {
      "id": "savings_tip",
      "type": "tip",
      "category": "savings",
      "title": "Savings Opportunity",
      "message": "You spent ₹2,000 on food delivery this week. Cooking at home could save ₹1,500/month.",
      "actionText": "See Details",
      "actionType": "info",
      "priority": "low"
    }
  ]
}
```

**Nudge Types:**
| Type | Description |
|------|-------------|
| warning | Alerts about budget/spending issues |
| celebration | Positive progress updates |
| tip | Helpful financial suggestions |
| reminder | Action reminders |

**Priority Levels:** `high`, `medium`, `low`

---

### Get Financial Health Score

Returns overall financial health assessment.

```
GET /api/predictions/health-score
```

**Authentication Required:** Yes

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "overallScore": 72,
    "breakdown": {
      "savingsScore": 85,
      "diversityScore": 60,
      "consistencyScore": 70
    },
    "metrics": {
      "savingsRate": 42,
      "totalIncome": 150000,
      "totalExpense": 87000,
      "netSavings": 63000,
      "categoriesUsed": 6
    },
    "recommendations": [
      "Great savings rate! Consider investing some of your savings.",
      "Track more expense categories for better insights.",
      "Set up a budget for your top spending category."
    ]
  }
}
```

**Score Breakdown:**
- **savingsScore (50% weight):** Based on savings rate (20% savings = 40 score, 50% = 100)
- **diversityScore (20% weight):** Based on expense category usage (10 categories = 100)
- **consistencyScore (30% weight):** Based on regular income tracking

---

## Analytics

### Get Dashboard

Returns financial dashboard summary.

```
GET /api/analytics/dashboard
```

**Authentication Required:** Yes

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "summary": {
      "totalIncome": 50000,
      "totalExpense": 15000,
      "balance": 35000,
      "savingsRate": 70
    },
    "recentTransactions": [...],
    "topCategories": [...],
    "goalProgress": [...],
    "monthlyComparison": {
      "incomeChange": 10,
      "expenseChange": -5
    }
  }
}
```

---

### Get Spending Trends

Returns spending trends over time.

```
GET /api/analytics/trends
```

**Authentication Required:** Yes

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| period | number | 30 | Days to analyze |

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "daily": [...],
    "categoryTrends": [...],
    "averageDaily": 500,
    "highestDay": {
      "date": "2025-11-15",
      "amount": 2500
    }
  }
}
```

---

### Get Income Analysis

Returns income pattern analysis.

```
GET /api/analytics/income
```

**Authentication Required:** Yes

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "totalIncome": 150000,
    "averageMonthly": 50000,
    "sources": [...],
    "patterns": {
      "bestDays": ["Friday", "Saturday"],
      "bestWeek": 3,
      "consistency": 75
    }
  }
}
```

---

## Coaching

All coaching endpoints provide AI-powered financial insights and advice.

### Get Financial Advice

Returns basic AI-generated personalized advice.

```
POST /api/coaching/advice
```

**Authentication Required:** Yes

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "advice": [
      {
        "type": "warning",
        "category": "savings",
        "title": "Low Savings Rate",
        "message": "Your savings rate is 5.2%. Try to save at least 20% of your income.",
        "actionable": "Set up automatic transfers to savings after each paycheck."
      },
      {
        "type": "info",
        "category": "spending",
        "title": "High Spending in One Category",
        "message": "45.3% of your expenses are in food. Consider reviewing these expenses.",
        "actionable": "Look for ways to reduce food costs."
      }
    ],
    "summary": {
      "income": 50000,
      "expenses": 47400,
      "savingsRate": 5.2,
      "activeGoals": 3
    }
  }
}
```

---

### Get Spending Insights

Returns detailed spending analysis.

```
GET /api/coaching/insights
```

**Authentication Required:** Yes

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| period | number | 30 | Days to analyze |

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "totalSpent": 47400,
    "averageTransaction": 1580,
    "mostExpensiveDay": "2025-11-15",
    "mostExpensiveCategory": "Food",
    "unusualSpending": []
  }
}
```

---

### Get AI Insights

Returns comprehensive AI-powered coaching insights.

```
GET /api/coaching/ai-insights
```

**Authentication Required:** Yes

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "insights": [
      {
        "id": "concentrated_spending",
        "type": "warning",
        "category": "spending",
        "priority": "medium",
        "title": "High Food Spending",
        "message": "45% of spending goes to food.",
        "detailedExplanation": "Spending ₹21,330 on food is a significant portion of your budget.",
        "actionItems": [
          "Review food expenses for potential cuts",
          "Look for alternatives or discounts",
          "Set a budget limit for this category"
        ],
        "potentialSavings": 3200
      },
      {
        "id": "excellent_savings",
        "type": "celebration",
        "category": "savings",
        "priority": "low",
        "title": "🎉 Excellent Saver!",
        "message": "You're saving 25.3% of your income. Outstanding!",
        "actionItems": [
          "Research SIP options",
          "Consider opening a PPF account",
          "Look into index funds"
        ]
      }
    ],
    "generatedAt": "2025-11-28T21:26:58.575Z",
    "totalInsights": 5,
    "highPriority": 1
  }
}
```

**Insight Types:** `tip`, `warning`, `celebration`, `action`, `education`

**Categories:** `savings`, `spending`, `income`, `goals`, `budgets`, `habits`, `general`

**Priorities:** `high`, `medium`, `low`

---

### Get Financial Profile

Returns a complete financial profile with behavioral patterns.

```
GET /api/coaching/profile
```

**Authentication Required:** Yes

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "userId": 1,
    "income": {
      "total": 150000,
      "average": 50000,
      "sources": [
        { "category": "transport", "amount": 120000 },
        { "category": "freelance", "amount": 30000 }
      ],
      "volatility": 25.5,
      "trend": "increasing"
    },
    "expenses": {
      "total": 112500,
      "average": 3750,
      "byCategory": [
        { "category": "food", "amount": 45000, "percentage": 40 },
        { "category": "transport", "amount": 22500, "percentage": 20 },
        { "category": "entertainment", "amount": 18000, "percentage": 16 }
      ],
      "trend": "stable"
    },
    "savings": {
      "rate": 25,
      "netAmount": 37500,
      "trend": "improving"
    },
    "goals": {
      "total": 3,
      "active": 2,
      "completed": 1,
      "atRisk": 0
    },
    "budgets": {
      "total": 4,
      "onTrack": 3,
      "exceeded": 1
    },
    "behaviorPatterns": {
      "spendingDays": ["Saturday", "Sunday"],
      "impulsePurchases": 5,
      "recurringExpenses": [
        { "description": "Netflix", "amount": 499, "frequency": "monthly" },
        { "description": "Gym", "amount": 1500, "frequency": "monthly" }
      ]
    }
  }
}
```

**Trend Values:** `increasing`, `decreasing`, `stable`

**Savings Trend:** `improving`, `declining`, `stable`

---

### Get Weekly Summary

Returns a weekly financial summary with comparisons.

```
GET /api/coaching/weekly-summary
```

**Authentication Required:** Yes

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "period": "22/11/2025 - 29/11/2025",
    "income": 15000,
    "expenses": 8750,
    "savings": 6250,
    "topCategories": [
      { "category": "food", "amount": 3500 },
      { "category": "transport", "amount": 2100 },
      { "category": "entertainment", "amount": 1500 }
    ],
    "comparison": {
      "incomeChange": 12.5,
      "expenseChange": -5.2
    },
    "highlights": [
      "💚 Positive cash flow: ₹6250 saved",
      "📉 Expenses decreased by 5.2%",
      "🎯 On track for monthly savings goal"
    ]
  }
}
```

---

### Get Action Plan

Returns a prioritized action plan based on financial profile.

```
GET /api/coaching/action-plan
```

**Authentication Required:** Yes

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "immediateActions": [
      {
        "action": "Review food expenses for potential cuts",
        "reason": "45% of spending goes to food.",
        "impact": "Could save ₹3200/month"
      },
      {
        "action": "Set up an emergency fund",
        "reason": "Variable income detected.",
        "impact": "Improve financial security"
      }
    ],
    "weeklyGoals": [
      {
        "goal": "Reduce food spending by 10%",
        "target": "Keep under ₹9100 this week"
      },
      {
        "goal": "Track all expenses this week",
        "target": "Log every purchase, no matter how small"
      }
    ],
    "monthlyTargets": [
      {
        "target": "Savings Rate",
        "metric": "20%",
        "current": "25.3%"
      },
      {
        "target": "Goal Progress",
        "metric": "On track",
        "current": "All on track"
      }
    ],
    "learningResources": [
      {
        "topic": "Managing Variable Income",
        "reason": "Your income varies significantly"
      },
      {
        "topic": "Introduction to Investing",
        "reason": "You have a good savings rate - time to grow your money"
      }
    ]
  }
}
```

---

### Chat with Coach

Interactive AI coaching with contextual responses.

```
POST /api/coaching/chat
```

**Authentication Required:** Yes

**Request Body:**
```json
{
  "message": "How can I save more?"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| message | string | Yes | User's question or topic |

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "message": "Your current savings rate is 15.2%. I recommend aiming for at least 20%. Start by tracking your top expense category (food) and look for ways to reduce it by 10%.",
    "context": {
      "savingsRate": "15.2",
      "topCategory": "food",
      "activeGoals": 2
    }
  }
}
```

**Supported Topics:**
- `save`/`saving` - Savings advice
- `budget`/`budgeting` - Budget recommendations  
- `goal`/`goals` - Goal progress and tips
- `spend`/`expense` - Spending analysis
- `income` - Income pattern insights
- `help`/`tip` - Random helpful tip

**Example Interactions:**
```
"How can I save more?"
"Help me budget"
"How are my goals progressing?"
"Analyze my spending"
"My income is variable"
"Give me a tip"
```

---

### Create Smart Notification

Creates a personalized notification.

```
POST /api/coaching/notify
```

**Authentication Required:** Yes

**Request Body:**
```json
{
  "title": "Weekly Savings Reminder",
  "message": "You've saved ₹6,250 this week!",
  "type": "success",
  "category": "coaching"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| title | string | Yes | Notification title |
| message | string | Yes | Notification content |
| type | string | Yes | info, warning, success, alert |
| category | string | Yes | transaction, goal, coaching, system |

**Success Response (201):**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "userId": 1,
    "title": "Weekly Savings Reminder",
    "message": "You've saved ₹6,250 this week!",
    "type": "success",
    "category": "coaching",
    "isRead": false,
    "createdAt": "2025-11-28T21:30:00Z",
    "updatedAt": "2025-11-28T21:30:00Z"
  }
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request - Invalid input |
| 401 | Unauthorized - Invalid/missing token |
| 404 | Not Found - Resource doesn't exist |
| 500 | Internal Server Error |

---


## Rate Limits

Currently no rate limiting is implemented. For production:
- Authentication endpoints: 5 requests/minute
- Other endpoints: 100 requests/minute

---

## Changelog

### v1.1.0 (November 2025)
- Enhanced AI coaching system
  - Comprehensive financial profile building
  - AI-powered insights with actionable recommendations
  - Weekly summary reports
  - Personalized action plans
  - Interactive chat with coach
- Improved coaching endpoints (8 total)
- Added behavioral pattern detection
- Enhanced investment recommendations

### v1.0.0 (November 2025)
- Initial release with core features
- Authentication, Transactions, Goals
- Budgets, Achievements, Predictions
- SMS bulk import with auto-categorization
- Financial health scoring
- Personalized nudges

---

## Support

For issues or questions, contact the development team or open an issue on GitHub.
