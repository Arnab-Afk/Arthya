# Migration from MongoDB to PostgreSQL

## Overview
Successfully migrated the Arthya backend from MongoDB with Mongoose to PostgreSQL with Sequelize ORM.

## Changes Made

### 1. Dependencies
**Removed:**
- `mongoose` - MongoDB ODM

**Added:**
- `pg` - PostgreSQL driver
- `sequelize` - ORM for PostgreSQL
- `sequelize-typescript` - TypeScript decorators for Sequelize
- `reflect-metadata` - Required for TypeScript decorators

### 2. Database Configuration
**File:** `src/config/database.ts`

**Changes:**
- Replaced MongoDB connection with PostgreSQL Sequelize instance
- Added connection pooling configuration
- Auto-sync models in development mode
- Export both `connectDB` function and `sequelize` instance

### 3. Environment Variables
**File:** `.env.example`

**Replaced:**
```env
MONGODB_URI=mongodb://localhost:27017/arthya
```

**With:**
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=arthya
DB_USER=postgres
DB_PASSWORD=your-database-password
```

### 4. Models Converted

All models converted from Mongoose schemas to Sequelize classes with TypeScript decorators:

#### User Model (`src/models/User.ts`)
- Uses `@Table`, `@Column` decorators
- Password hashing moved to table-level hooks
- Changed `_id` to auto-incrementing `id`
- Implements `comparePassword` method

#### Transaction Model (`src/models/Transaction.ts`)
- Added foreign key relationship to User
- Uses DECIMAL(10,2) for amounts
- JSONB for metadata storage
- Maintains indexes for performance

#### Goal Model (`src/models/Goal.ts`)
- Foreign key to User model
- JSONB for milestones array
- Goal status auto-completion logic preserved

#### Card Model (`src/models/Card.ts`)
- Foreign key to User model
- DECIMAL for balance and creditLimit
- Boolean flags for isActive and isPrimary

#### Notification Model (`src/models/Notification.ts`)
- Foreign key to User model
- JSONB for flexible metadata
- Indexed for efficient querying

### 5. Controllers Updated

All controllers updated to use Sequelize query methods:

#### Auth Controller (`src/controllers/authController.ts`)
- `User.findOne({ email })` → `User.findOne({ where: { email } })`
- `User.findById(id)` → `User.findByPk(id)`
- `User.findByIdAndUpdate()` → `user.update()` + `user.save()`
- JWT token now uses numeric `id` instead of string `_id`

#### Transaction Controller (`src/controllers/transactionController.ts`)
- Added `Op` operators from Sequelize for date ranges
- `Transaction.find()` → `Transaction.findAndCountAll()`
- `Transaction.countDocuments()` → included in `findAndCountAll`
- Aggregations using `sequelize.fn()` and `sequelize.col()`
- `transaction.deleteOne()` → `transaction.destroy()`

#### Goal Controller (`src/controllers/goalController.ts`)
- `Goal.find()` → `Goal.findAll()`
- `Goal.findById()` → `Goal.findOne({ where: { id } })`
- `goal.save()` for updates preserved
- Auto-completion logic for goals maintained

#### Analytics Controller (`src/controllers/analyticsController.ts`)
- Complex MongoDB aggregations converted to Sequelize queries
- Uses `sequelize.fn('SUM')`, `sequelize.fn('COUNT')`, etc.
- Date functions using `sequelize.fn('EXTRACT')`
- Grouping and ordering using Sequelize syntax

#### Coaching Controller (`src/controllers/coachingController.ts`)
- Date range queries using `Op.gte` operator
- Array filtering and reduce operations updated for numeric amounts
- All MongoDB queries converted to Sequelize

### 6. Middleware Updated

**File:** `src/middleware/auth.ts`
- Changed user type from `IUser` interface to `User` class
- JWT decode expects numeric `id` instead of string
- `User.findById()` → `User.findByPk()`

### 7. Server Configuration

**File:** `src/server.ts`
- Import changed: `import connectDB from` → `import { connectDB } from`

## Setup Instructions

### 1. Install PostgreSQL
```bash
# macOS
brew install postgresql@15
brew services start postgresql@15

# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql

# Windows
# Download installer from postgresql.org
```

### 2. Create Database
```bash
# Login to PostgreSQL
psql postgres

# Create database
CREATE DATABASE arthya;

# Create user (if needed)
CREATE USER your_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE arthya TO your_user;
```

### 3. Configure Environment
Create `.env` file in backend directory:
```env
PORT=3000
NODE_ENV=development

# PostgreSQL Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=arthya
DB_USER=postgres
DB_PASSWORD=your_password

JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
JWT_EXPIRE=7d

ML_SERVICE_URL=http://localhost:8000
CLIENT_URL=http://localhost:8081
```

### 4. Install Dependencies
```bash
cd backend
npm install
```

### 5. Run the Server
```bash
# Development
npm run dev

# Production
npm run build
npm start
```

The application will automatically create all tables on first run in development mode.

## Key Differences

### ID Fields
- **MongoDB:** Uses `_id` (ObjectId string)
- **PostgreSQL:** Uses `id` (auto-incrementing integer)

### Queries
- **MongoDB:** `Model.find({ field: value })`
- **Sequelize:** `Model.findAll({ where: { field: value } })`

### Updates
- **MongoDB:** `Model.findByIdAndUpdate(id, data)`
- **Sequelize:** `model.update(data)` then `model.save()`

### Aggregations
- **MongoDB:** `Model.aggregate([{ $group: ... }])`
- **Sequelize:** `Model.findAll({ attributes: [sequelize.fn(...)], group: [...] })`

### Relationships
- **MongoDB:** Manual population with `ref`
- **Sequelize:** Automatic with `@ForeignKey` and `@BelongsTo`

## Data Migration

To migrate existing MongoDB data to PostgreSQL:

1. Export from MongoDB:
```bash
mongoexport --db=arthya --collection=users --out=users.json
mongoexport --db=arthya --collection=transactions --out=transactions.json
# ... repeat for all collections
```

2. Transform and import to PostgreSQL (create custom script or use ETL tools)

## Testing

After migration:
1. Test user registration and login
2. Verify all CRUD operations for transactions
3. Check goal creation and progress updates
4. Test analytics dashboard endpoints
5. Verify coaching advice generation

## Performance Considerations

- PostgreSQL uses connection pooling (max: 5 connections)
- Indexes are automatically created based on model decorators
- JSONB fields provide efficient JSON storage
- Sequelize auto-syncs schema in development mode

## Rollback Plan

If issues occur, the old MongoDB controllers are backed up with `.old.ts` extension (already removed after successful migration). To rollback:

1. Stop the server
2. Reinstall mongoose: `npm install mongoose`
3. Restore old database config and models
4. Update environment variables back to MongoDB

## Status

✅ Migration Complete
✅ Build Successful
✅ All TypeScript Errors Resolved
✅ Ready for Testing
