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
export type OccupationType = 'driver' | 'delivery' | 'freelancer' | 'domestic' | 'vendor' | 'daily_wage' | 'other';

export interface User {
  id: number;
  name: string;
  email: string;
  phone?: string;
  occupation?: OccupationType;
  monthlyIncome?: number;
  createdAt: string;
  updatedAt?: string;
}

export interface LoginResponse {
  id: number;
  name: string;
  email: string;
  occupation?: string;
  token: string;
}

// Transaction Types
export interface Transaction {
  id: number;
  userId: number;
  type: 'income' | 'expense' | 'transfer';
  category: string;
  amount: string | number;
  description?: string;
  date: string;
  recipient?: string;
  status: 'completed' | 'pending' | 'failed';
  source?: string;
  notes?: string;
  createdAt: string;
  updatedAt?: string;
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

export interface CategorySpending {
  category: string;
  total: number;
  count: number;
  percentage: string;
}

export interface SpendingByCategoryResponse {
  categories: CategorySpending[];
  totalSpending: number;
}

export interface MonthlyTrend {
  month: string;
  income: number;
  expense: number;
  savings: number;
  savingsRate: number;
}

// Goal Types
export interface Goal {
  id: number;
  userId: number;
  name: string;
  description?: string;
  targetAmount: string | number;
  currentAmount: string | number;
  deadline?: string;
  category?: string;
  priority?: 'low' | 'medium' | 'high';
  status: 'active' | 'completed' | 'cancelled';
  progress: number;
  createdAt: string;
  updatedAt?: string;
}

// Budget Types
export interface Budget {
  id: number;
  userId: number;
  category: string;
  amount: string | number;
  period: 'weekly' | 'monthly' | 'yearly';
  startDate: string;
  endDate?: string;
  isActive: boolean;
  alertThreshold: number;
  spent: number;
  remaining: number;
  percentUsed: number;
  notes?: string;
  createdAt?: string;
  updatedAt?: string;
}

export interface BudgetSummary {
  totalBudget: number;
  totalSpent: number;
  totalRemaining: number;
  overallPercentUsed: number;
  budgetCount: number;
  categoryBreakdown: {
    category: string;
    budgeted: number;
    spent: number;
    remaining: number;
    percentUsed: number;
    status: 'good' | 'warning' | 'exceeded';
  }[];
}

// Achievement Types
export interface Achievement {
  type: string;
  title: string;
  description: string;
  icon: string;
  category: string;
  unlocked: boolean;
  unlockedAt?: string;
  progress: number;
}

export interface AchievementsResponse {
  achievements: Achievement[];
  totalUnlocked: number;
  totalAchievements: number;
  completionPercentage: number;
}

export interface AchievementProgress {
  transactions: { current: number; next_milestone: number };
  savings: { current: number; next_milestone: number };
  categories: { current: number; target: number };
  goals: { created: number; completed: number };
}

// Prediction Types
export interface IncomePrediction {
  predictedAmount: number;
  confidence: number;
  trend: 'increasing' | 'decreasing' | 'stable';
  seasonalPattern: string;
  volatility: number;
  recommendations: string[];
}

export interface ExpensePrediction {
  predictions: { month: string; predicted: number }[];
  categoryPredictions: { category: string; avgMonthly: number; trend: string }[];
  historicalAverage: number;
  dataPointsUsed: number;
}

export interface CashFlowProjection {
  currentAverages: {
    income: number;
    expense: number;
    netMonthly: number;
  };
  projections: {
    month: string;
    projectedIncome: number;
    projectedExpense: number;
    netCashFlow: number;
    runningBalance: number;
  }[];
  savingsRate: number;
}

export interface Nudge {
  id: string;
  type: 'warning' | 'celebration' | 'tip' | 'reminder';
  category: string;
  title: string;
  message: string;
  actionText?: string;
  actionType?: string;
  priority: 'high' | 'medium' | 'low';
}

export interface FinancialHealthScore {
  overallScore: number;
  breakdown: {
    savingsScore: number;
    diversityScore: number;
    consistencyScore: number;
  };
  metrics: {
    savingsRate: number;
    totalIncome: number;
    totalExpense: number;
    netSavings: number;
    categoriesUsed: number;
  };
  recommendations: string[];
}

// Dashboard Types
export interface DashboardData {
  summary: {
    totalIncome: number;
    totalExpense: number;
    balance: number;
    savingsRate: number;
  };
  recentTransactions: Transaction[];
  topCategories: CategorySpending[];
  goalProgress: Goal[];
  monthlyComparison: {
    incomeChange: number;
    expenseChange: number;
  };
}

// Analytics Types
export interface TrendsData {
  daily: { date: string; amount: number }[];
  categoryTrends: CategorySpending[];
  averageDaily: number;
  highestDay: { date: string; amount: number };
}

export interface IncomeAnalysis {
  totalIncome: number;
  averageMonthly: number;
  sources: { category: string; amount: number }[];
  patterns: {
    bestDays: string[];
    bestWeek: number;
    consistency: number;
  };
}

// Coaching Types
export interface FinancialAdvice {
  type: 'info' | 'warning' | 'success' | 'alert' | 'tip' | 'celebration' | 'action' | 'education';
  category: 'savings' | 'spending' | 'goals' | 'income' | 'budgets' | 'habits' | 'general';
  title: string;
  message: string;
  actionable?: string;
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

export interface AIInsight {
  id: string;
  type: 'tip' | 'warning' | 'celebration' | 'action' | 'education';
  category: 'savings' | 'spending' | 'income' | 'goals' | 'budgets' | 'habits' | 'general';
  priority: 'high' | 'medium' | 'low';
  title: string;
  message: string;
  detailedExplanation?: string;
  actionItems?: string[];
  potentialSavings?: number;
}

export interface AIInsightsResponse {
  insights: AIInsight[];
  generatedAt: string;
  totalInsights: number;
  highPriority: number;
}

export interface FinancialProfile {
  userId: number;
  income: {
    total: number;
    average: number;
    sources: { category: string; amount: number }[];
    volatility: number;
    trend: 'increasing' | 'decreasing' | 'stable';
  };
  expenses: {
    total: number;
    average: number;
    byCategory: { category: string; amount: number; percentage: number }[];
    trend: 'increasing' | 'decreasing' | 'stable';
  };
  savings: {
    rate: number;
    netAmount: number;
    trend: 'improving' | 'declining' | 'stable';
  };
  goals: {
    total: number;
    active: number;
    completed: number;
    atRisk: number;
  };
  budgets: {
    total: number;
    onTrack: number;
    exceeded: number;
  };
  behaviorPatterns: {
    spendingDays: string[];
    impulsePurchases: number;
    recurringExpenses: { description: string; amount: number; frequency: string }[];
  };
}

export interface WeeklySummary {
  period: string;
  income: number;
  expenses: number;
  savings: number;
  topCategories: { category: string; amount: number }[];
  comparison: {
    incomeChange: number;
    expenseChange: number;
  };
  highlights: string[];
}

export interface ActionPlan {
  immediateActions: {
    action: string;
    reason: string;
    impact: string;
  }[];
  weeklyGoals: {
    goal: string;
    target: string;
  }[];
  monthlyTargets: {
    target: string;
    metric: string;
    current: string;
  }[];
  learningResources: {
    topic: string;
    reason: string;
  }[];
}

export interface ChatResponse {
  message: string;
  context: {
    savingsRate?: string;
    topCategory?: string;
    activeGoals?: number;
  };
}

// Notification Types
export interface Notification {
  id: number;
  userId: number;
  title: string;
  message: string;
  type: 'info' | 'warning' | 'success' | 'alert';
  category: 'transaction' | 'goal' | 'coaching' | 'system';
  isRead: boolean;
  createdAt: string;
  updatedAt: string;
}

// Request Types
export interface RegisterRequest {
  name: string;
  email: string;
  password: string;
  phone?: string;
  occupation?: OccupationType;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface CreateTransactionRequest {
  type: 'income' | 'expense' | 'transfer';
  amount: number;
  category: string;
  description?: string;
  date?: string;
  recipient?: string;
  status?: 'completed' | 'pending' | 'failed';
  notes?: string;
}

export interface BulkImportTransaction {
  type: 'income' | 'expense';
  amount: number;
  description: string;
  category?: string;
  date: string;
  source?: string;
  notes?: string;
}

export interface BulkImportResponse {
  imported: number;
  skipped: number;
  errors: string[];
  message: string;
}

export interface CreateGoalRequest {
  name: string;
  description?: string;
  targetAmount: number;
  currentAmount?: number;
  deadline?: string;
  category?: string;
  priority?: 'low' | 'medium' | 'high';
}

export interface CreateBudgetRequest {
  category: string;
  amount: number;
  period?: 'weekly' | 'monthly' | 'yearly';
  startDate?: string;
  endDate?: string;
  alertThreshold?: number;
  notes?: string;
}

export interface CreateNotificationRequest {
  title: string;
  message: string;
  type: 'info' | 'warning' | 'success' | 'alert';
  category: 'transaction' | 'goal' | 'coaching' | 'system';
}
