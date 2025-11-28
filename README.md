# Arthya 💰🤖

AI-powered autonomous financial coaching agent designed to provide personalized, proactive financial guidance to gig workers, informal sector employees, and everyday citizens. The agent adapts to real user behavior, spending patterns, and income variability to deliver contextual financial advice that helps users make smarter financial decisions.

## 🌟 Features

- 📱 **Beautiful Mobile App** - React Native/Expo app with modern UI
- 🔐 **Secure Authentication** - JWT-based user authentication
- 💸 **Transaction Tracking** - Track income, expenses, and transfers
- 🎯 **Goal Management** - Set and track financial goals with progress monitoring
- 📊 **Smart Analytics** - Comprehensive financial insights and trends
- 🤖 **AI Coaching** - Personalized financial advice based on spending patterns
- 💳 **Multi-Card Support** - Manage multiple payment cards
- 🔔 **Smart Notifications** - Proactive alerts and financial guidance
- 📈 **Income Analysis** - Special features for variable income (gig workers)

## 🏗️ Project Structure

```
Arthya/
├── backend/                 # Node.js/Express/TypeScript API
│   ├── src/
│   │   ├── controllers/    # Request handlers
│   │   ├── models/         # MongoDB models
│   │   ├── routes/         # API routes
│   │   ├── middleware/     # Auth & error handling
│   │   └── config/         # Database configuration
│   ├── dist/               # Compiled JavaScript
│   ├── README.md           # Backend documentation
│   ├── API_TESTING.md      # API testing guide
│   └── package.json
├── mobile/                  # React Native/Expo app
│   ├── app/                # App screens & navigation
│   ├── components/         # Reusable components
│   ├── constants/          # Theme & colors
│   └── package.json
├── ml/                      # Machine learning models
│   ├── main.py
│   └── dataGenerator.py
└── MOBILE_INTEGRATION.md   # Integration guide
```

## 🚀 Quick Start

### Prerequisites

- Node.js v18+
- MongoDB (local or Atlas)
- iOS Simulator or Android Emulator (for mobile)

### Backend Setup

```bash
# Navigate to backend
cd backend

# Install dependencies
npm install

# Start MongoDB (if local)
# mongod

# Start development server
npm run dev
```

Backend will run on `http://localhost:3000`

### Mobile App Setup

```bash
# Navigate to mobile
cd mobile

# Install dependencies
npm install

# Start Expo
npx expo start

# Press 'i' for iOS or 'a' for Android
```

## 📚 Documentation

- **[Backend Documentation](backend/README.md)** - Complete API reference
- **[API Testing Guide](backend/API_TESTING.md)** - Test endpoints with cURL/Postman
- **[Mobile Integration](MOBILE_INTEGRATION.md)** - Connect mobile app to backend
- **[Implementation Summary](backend/IMPLEMENTATION_SUMMARY.md)** - Complete feature list
- **[Docker Deployment](backend/DOCKER.md)** - Containerized deployment

## 🔌 API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user
- `GET /api/auth/me` - Get current user
- `PUT /api/auth/profile` - Update profile

### Transactions
- `GET /api/transactions` - List transactions
- `POST /api/transactions` - Create transaction
- `GET /api/transactions/summary` - Get summary

### Goals
- `GET /api/goals` - List goals
- `POST /api/goals` - Create goal
- `PUT /api/goals/:id/progress` - Update progress

### Analytics
- `GET /api/analytics/dashboard` - Full dashboard
- `GET /api/analytics/trends` - Spending trends
- `GET /api/analytics/income` - Income analysis

### AI Coaching
- `POST /api/coaching/advice` - Get personalized advice
- `GET /api/coaching/insights` - Get spending insights

**Total: 22 API endpoints**

## 🛠️ Tech Stack

### Backend
- **Runtime**: Node.js
- **Framework**: Express.js
- **Language**: TypeScript
- **Database**: MongoDB + Mongoose
- **Authentication**: JWT
- **Security**: Helmet, bcrypt, CORS

### Mobile
- **Framework**: React Native + Expo
- **Language**: TypeScript
- **UI**: Custom components with animations
- **Navigation**: Expo Router
- **State**: React Hooks

### ML (Planned)
- **Language**: Python
- **Framework**: TensorFlow/PyTorch
- **Analysis**: Pandas, NumPy

## 🎯 Key Features for Gig Workers

1. **Variable Income Detection** - Identifies irregular income patterns
2. **Smart Savings Recommendations** - Advice for variable income
3. **Emergency Fund Guidance** - 3-6 months expense coverage
4. **Flexible Categorization** - Track different income sources
5. **Occupation Types** - Driver, Freelancer, Hybrid, Other

## 🧪 Testing

### Test Backend API

```bash
# Health check
curl http://localhost:3000/health

# Register user
curl -X POST http://localhost:3000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"name":"Test User","email":"test@example.com","password":"test123","occupation":"driver"}'
```

See [API_TESTING.md](backend/API_TESTING.md) for complete examples.

## 🐳 Docker Deployment

```bash
cd backend

# Start MongoDB + Backend
docker-compose up -d

# View logs
docker-compose logs -f backend
```

See [DOCKER.md](backend/DOCKER.md) for details.

## 📊 Database Schema

### User
- Name, email, password (hashed)
- Occupation (driver/freelancer/hybrid/other)
- Avatar, phone

### Transaction
- Type (income/expense/transfer)
- Category, amount, description
- Date, status, metadata

### Goal
- Title, target/current amount
- Category, deadline, status
- Milestones with progress tracking

### Card
- Card type, number, balance
- Credit limit, status

## 🔒 Security

- ✅ Password hashing (bcrypt with 10 salt rounds)
- ✅ JWT token-based authentication
- ✅ Protected routes with middleware
- ✅ HTTP security headers (Helmet)
- ✅ CORS configuration
- ✅ Input validation
- ✅ Error handling

## 📱 Mobile Features

- ✅ Beautiful dashboard with animations
- ✅ Transaction tracking
- ✅ Goal visualization with progress bars
- ✅ Multiple card support
- ✅ Quick transfer to contacts
- ✅ Financial summaries (income/expense/savings)
- ✅ Dark-themed modern UI

## 🚀 Deployment

### Backend Options
1. **Docker Compose** (Recommended)
2. **Heroku** - Node.js + MongoDB Atlas
3. **DigitalOcean** - App Platform or Droplet
4. **AWS** - EC2 or Elastic Beanstalk
5. **Azure** - App Service

### Mobile Options
1. **Expo EAS** - Managed builds
2. **TestFlight** (iOS) - Beta testing
3. **Google Play** (Android) - Production

## 🛣️ Roadmap

### Phase 1 - MVP ✅
- [x] Backend API with authentication
- [x] Transaction management
- [x] Goal tracking
- [x] Basic analytics
- [x] AI coaching foundation
- [x] Mobile UI

### Phase 2 - Enhancement 🚧
- [ ] ML model integration
- [ ] Real-time notifications
- [ ] Receipt scanning
- [ ] Data export (CSV/PDF)
- [ ] Two-factor authentication

### Phase 3 - Scale 📋
- [ ] Multi-language support
- [ ] Advanced analytics
- [ ] Investment tracking
- [ ] Bill reminders
- [ ] Expense splitting

## 👥 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🤝 Support

For issues or questions:
- Check [Backend README](backend/README.md)
- Review [API Testing Guide](backend/API_TESTING.md)
- See [Mobile Integration Guide](MOBILE_INTEGRATION.md)

## ✨ Acknowledgments

Built with ❤️ for gig workers and everyone seeking better financial management.

---

**Status**: Backend Complete ✅ | Mobile UI Complete ✅ | Integration Ready 🚀
