// API Response Types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  count?: number;
  total?: number;
  page?: number;
  pages?: number;
}

// User Types
export interface User {
  _id: string;
  name: string;
  email: string;
  phone?: string;
  occupation: 'driver' | 'freelancer' | 'hybrid' | 'other';
  avatar?: string;
  createdAt: string;
  updatedAt: string;
}

export interface LoginResponse {
  id: string;
  name: string;
  email: string;
  occupation: string;
  token: string;
}

// Transaction Types
export interface Transaction {
  _id: string;
  userId: string;
  type: 'income' | 'expense' | 'transfer';
  category: string;
  amount: number;
  description?: string;
  date: string;
  recipient?: string;
  status: 'completed' | 'pending' | 'failed';
  metadata?: {
    location?: string;
    paymentMethod?: string;
    merchantName?: string;
  };
  createdAt: string;
  updatedAt: string;
}

export interface TransactionSummary {
  income: number;
  expense: number;
  transfer: number;
  incomeCount: number;
  expenseCount: number;
  transferCount: number;
  savings: number;
}

// Goal Types
export interface Goal {
  _id: string;
  userId: string;
  title: string;
  description?: string;
  targetAmount: number;
  currentAmount: number;
  icon?: string;
  category: 'savings' | 'purchase' | 'investment' | 'debt' | 'other';
  deadline?: string;
  status: 'active' | 'completed' | 'cancelled';
  milestones?: {
    amount: number;
    date: string;
    achieved: boolean;
  }[];
  createdAt: string;
  updatedAt: string;
}

// Card Types
export interface Card {
  _id: string;
  userId: string;
  cardNumber: string;
  cardType: 'paypal' | 'payeer' | 'debit' | 'credit';
  cardholderName: string;
  expiryDate: string;
  balance: number;
  creditLimit?: number;
  isActive: boolean;
  isPrimary: boolean;
  createdAt: string;
  updatedAt: string;
}

// Dashboard Types
export interface DashboardData {
  summary: {
    income: number;
    expense: number;
    savings: number;
    availableBalance: number;
    creditLimit: number;
  };
  activeGoals: Goal[];
  recentTransactions: Transaction[];
  cards: Card[];
}

// Analytics Types
export interface SpendingTrend {
  _id: {
    date: string;
    type: string;
  };
  total: number;
}

export interface CategoryBreakdown {
  _id: string;
  total: number;
  count: number;
}

export interface TrendsData {
  trends: SpendingTrend[];
  categoryBreakdown: CategoryBreakdown[];
}

// Coaching Types
export interface FinancialAdvice {
  type: 'info' | 'warning' | 'success' | 'alert';
  category: 'savings' | 'spending' | 'goals' | 'income';
  title: string;
  message: string;
  actionable: string;
}

export interface CoachingAdviceResponse {
  advice: FinancialAdvice[];
  summary: {
    income: number;
    expenses: number;
    savingsRate: number;
    activeGoals: number;
  };
}

// Request Types
export interface RegisterRequest {
  name: string;
  email: string;
  password: string;
  phone?: string;
  occupation: 'driver' | 'freelancer' | 'hybrid' | 'other';
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface CreateTransactionRequest {
  type: 'income' | 'expense' | 'transfer';
  category: string;
  amount: number;
  description?: string;
  date?: string;
  recipient?: string;
}

export interface CreateGoalRequest {
  title: string;
  description?: string;
  targetAmount: number;
  currentAmount?: number;
  icon?: string;
  category: 'savings' | 'purchase' | 'investment' | 'debt' | 'other';
  deadline?: string;
}
