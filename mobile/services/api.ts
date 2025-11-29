import axios, { AxiosInstance, AxiosError } from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Platform } from 'react-native';
import {
  ApiResponse,
  LoginRequest,
  LoginResponse,
  RegisterRequest,
  User,
  Transaction,
  TransactionSummary,
  Goal,
  DashboardData,
  TrendsData,
  CoachingAdviceResponse,
  CreateTransactionRequest,
  CreateGoalRequest,
  Budget,
  BudgetSummary,
  CreateBudgetRequest,
  AchievementsResponse,
  AchievementProgress,
  IncomePrediction,
  ExpensePrediction,
  CashFlowProjection,
  Nudge,
  FinancialHealthScore,
  SpendingByCategoryResponse,
  MonthlyTrend,
  IncomeAnalysis,
  AIInsightsResponse,
  FinancialProfile,
  WeeklySummary,
  ActionPlan,
  ChatResponse,
  BulkImportTransaction,
  BulkImportResponse,
  Notification,
  CreateNotificationRequest,
} from '@/types/api';

// Change this to your computer's IP address when testing on physical device
// On iOS simulator, localhost works fine
// On Android emulator, use 10.0.2.2 instead of localhost

const getApiUrl = () => {
  if (!__DEV__) {
    return 'https://arthyapi.ishikabhoyar.tech/api';
  }
  
  // For Android emulator, use 10.0.2.2
  // For physical device, use your computer's local IP
  // For iOS simulator, localhost works
  if (Platform.OS === 'android') {
    // Use your local IP for physical device, or 10.0.2.2 for emulator
    return 'https://arthyapi.ishikabhoyar.tech/api';
  }
  
  return 'https://arthyapi.ishikabhoyar.tech/api';
};

const API_URL = getApiUrl();

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_URL,
      timeout: 30000, // 30 seconds for deployed server
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor - Add token to requests
    this.client.interceptors.request.use(
      async (config) => {
        try {
          const token = await AsyncStorage.getItem('token');
          if (token) {
            config.headers.Authorization = `Bearer ${token}`;
          }
        } catch (error) {
          console.error('Error getting token:', error);
        }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor - Handle responses and errors
    this.client.interceptors.response.use(
      (response) => response.data,
      async (error: AxiosError) => {
        if (error.response?.status === 401) {
          // Token expired or invalid - clear token
          await AsyncStorage.removeItem('token');
          await AsyncStorage.removeItem('user');
        }
        
        const errorMessage = 
          (error.response?.data as any)?.message || 
          error.message || 
          'Something went wrong';
        
        return Promise.reject(new Error(errorMessage));
      }
    );
  }

  // Authentication Methods
  async register(data: RegisterRequest): Promise<ApiResponse<LoginResponse>> {
    return this.client.post('/auth/register', data);
  }

  async login(credentials: LoginRequest): Promise<ApiResponse<LoginResponse>> {
    const response: ApiResponse<LoginResponse> = await this.client.post('/auth/login', credentials);
    
    // Store token if login successful
    if (response.success && response.data?.token) {
      await AsyncStorage.setItem('token', response.data.token);
      await AsyncStorage.setItem('user', JSON.stringify(response.data));
    }
    
    return response;
  }

  async logout(): Promise<void> {
    await AsyncStorage.removeItem('token');
    await AsyncStorage.removeItem('user');
  }

  async getProfile(): Promise<ApiResponse<User>> {
    return this.client.get('/auth/me');
  }

  async updateProfile(data: Partial<User>): Promise<ApiResponse<User>> {
    return this.client.put('/auth/profile', data);
  }

  // Transaction Methods
  async getTransactions(params?: {
    type?: string;
    startDate?: string;
    endDate?: string;
    category?: string;
    limit?: number;
    page?: number;
  }): Promise<ApiResponse<Transaction[]>> {
    return this.client.get('/transactions', { params });
  }

  async getTransaction(id: string): Promise<ApiResponse<Transaction>> {
    return this.client.get(`/transactions/${id}`);
  }

  async createTransaction(data: CreateTransactionRequest): Promise<ApiResponse<Transaction>> {
    return this.client.post('/transactions', data);
  }

  async updateTransaction(id: string, data: Partial<CreateTransactionRequest>): Promise<ApiResponse<Transaction>> {
    return this.client.put(`/transactions/${id}`, data);
  }

  async deleteTransaction(id: string): Promise<ApiResponse<void>> {
    return this.client.delete(`/transactions/${id}`);
  }

  async getTransactionSummary(startDate?: string, endDate?: string): Promise<ApiResponse<TransactionSummary>> {
    return this.client.get('/transactions/summary', {
      params: { startDate, endDate },
    });
  }

  // Goal Methods
  async getGoals(status?: string): Promise<ApiResponse<Goal[]>> {
    return this.client.get('/goals', { params: { status } });
  }

  async getGoal(id: string): Promise<ApiResponse<Goal>> {
    return this.client.get(`/goals/${id}`);
  }

  async createGoal(data: CreateGoalRequest): Promise<ApiResponse<Goal>> {
    return this.client.post('/goals', data);
  }

  async updateGoal(id: string, data: Partial<CreateGoalRequest>): Promise<ApiResponse<Goal>> {
    return this.client.put(`/goals/${id}`, data);
  }

  async updateGoalProgress(id: string, amount: number): Promise<ApiResponse<Goal>> {
    return this.client.put(`/goals/${id}/progress`, { amount });
  }

  async deleteGoal(id: string): Promise<ApiResponse<void>> {
    return this.client.delete(`/goals/${id}`);
  }

  // Analytics Methods
  async getDashboard(): Promise<ApiResponse<DashboardData>> {
    return this.client.get('/analytics/dashboard');
  }

  async getSpendingTrends(period: number = 30): Promise<ApiResponse<TrendsData>> {
    return this.client.get('/analytics/trends', { params: { period } });
  }

  async getIncomeAnalysis(): Promise<ApiResponse<IncomeAnalysis>> {
    return this.client.get('/analytics/income');
  }

  // Coaching Methods
  async getFinancialAdvice(): Promise<ApiResponse<CoachingAdviceResponse>> {
    return this.client.post('/coaching/advice');
  }

  async getSpendingInsights(period: number = 30): Promise<ApiResponse<{
    totalSpent: number;
    averageTransaction: number;
    mostExpensiveDay: string;
    mostExpensiveCategory: string;
    unusualSpending: any[];
  }>> {
    return this.client.get('/coaching/insights', { params: { period } });
  }

  async getAIInsights(): Promise<ApiResponse<AIInsightsResponse>> {
    return this.client.get('/coaching/ai-insights');
  }

  async getFinancialProfile(): Promise<ApiResponse<FinancialProfile>> {
    return this.client.get('/coaching/profile');
  }

  async getWeeklySummary(): Promise<ApiResponse<WeeklySummary>> {
    return this.client.get('/coaching/weekly-summary');
  }

  async getActionPlan(): Promise<ApiResponse<ActionPlan>> {
    return this.client.get('/coaching/action-plan');
  }

  async chatWithCoach(message: string, history?: { role: 'user' | 'model'; content: string }[]): Promise<ApiResponse<ChatResponse>> {
    return this.client.post('/coaching/chat', { message, history });
  }

  // Smart Notification
  async createNotification(data: CreateNotificationRequest): Promise<ApiResponse<Notification>> {
    return this.client.post('/coaching/notify', data);
  }

  // Gemini AI-powered endpoints
  async getGeminiAdvice(): Promise<ApiResponse<any>> {
    return this.client.get('/coaching/gemini-advice');
  }

  async getGeminiWeeklySummary(): Promise<ApiResponse<any>> {
    return this.client.get('/coaching/gemini-weekly');
  }

  async getSpendingAnalysis(days: number = 30): Promise<ApiResponse<any>> {
    return this.client.get('/coaching/spending-analysis', { params: { days } });
  }

  async getGoalAdvice(goalId: string): Promise<ApiResponse<any>> {
    return this.client.get(`/coaching/goal-advice/${goalId}`);
  }

  // Budget Methods
  async getBudgets(isActive?: boolean): Promise<ApiResponse<Budget[]>> {
    return this.client.get('/budgets', { params: { isActive } });
  }

  async getBudget(id: string): Promise<ApiResponse<Budget>> {
    return this.client.get(`/budgets/${id}`);
  }

  async createBudget(data: CreateBudgetRequest): Promise<ApiResponse<Budget>> {
    return this.client.post('/budgets', data);
  }

  async updateBudget(id: string, data: Partial<CreateBudgetRequest>): Promise<ApiResponse<Budget>> {
    return this.client.put(`/budgets/${id}`, data);
  }

  async deleteBudget(id: string): Promise<ApiResponse<void>> {
    return this.client.delete(`/budgets/${id}`);
  }

  async getBudgetSummary(): Promise<ApiResponse<BudgetSummary>> {
    return this.client.get('/budgets/summary');
  }

  // Achievement Methods
  async getAchievements(): Promise<ApiResponse<AchievementsResponse>> {
    return this.client.get('/achievements');
  }

  async checkAchievements(): Promise<ApiResponse<{
    newlyUnlocked: { id: number; type: string; title: string; description: string; icon: string; unlockedAt: string }[];
    message: string;
  }>> {
    return this.client.post('/achievements/check');
  }

  async getAchievementProgress(): Promise<ApiResponse<AchievementProgress>> {
    return this.client.get('/achievements/progress');
  }

  async getAchievementLeaderboard(): Promise<ApiResponse<{ userId: number; achievementCount: number }[]>> {
    return this.client.get('/achievements/leaderboard');
  }

  // Prediction Methods
  async getIncomePrediction(months?: number): Promise<ApiResponse<IncomePrediction>> {
    return this.client.get('/predictions/income', { params: { months } });
  }

  async getExpensePrediction(months?: number): Promise<ApiResponse<ExpensePrediction>> {
    return this.client.get('/predictions/expenses', { params: { months } });
  }

  async getCashFlowProjection(months?: number): Promise<ApiResponse<CashFlowProjection>> {
    return this.client.get('/predictions/cashflow', { params: { months } });
  }

  async getNudges(): Promise<ApiResponse<Nudge[]>> {
    return this.client.get('/predictions/nudges');
  }

  async getFinancialHealthScore(): Promise<ApiResponse<FinancialHealthScore>> {
    return this.client.get('/predictions/health-score');
  }

  // SMS Import Methods
  async bulkImportTransactions(transactions: BulkImportTransaction[]): Promise<ApiResponse<BulkImportResponse>> {
    return this.client.post('/transactions/bulk-import', { transactions });
  }

  async getSpendingByCategory(startDate?: string, endDate?: string): Promise<ApiResponse<SpendingByCategoryResponse>> {
    return this.client.get('/transactions/by-category', { params: { startDate, endDate } });
  }

  async getMonthlyTrends(months?: number): Promise<ApiResponse<MonthlyTrend[]>> {
    return this.client.get('/transactions/monthly-trends', { params: { months } });
  }

  // Health Check
  async healthCheck(): Promise<any> {
    return axios.get(`${API_URL.replace('/api', '')}/health`);
  }
}

export default new ApiService();
