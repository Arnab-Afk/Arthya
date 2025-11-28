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
} from '@/types/api';

// Change this to your computer's IP address when testing on physical device
// On iOS simulator, localhost works fine
// On Android emulator, use 10.0.2.2 instead of localhost

const getApiUrl = () => {
  if (!__DEV__) {
    return 'https://your-production-api.com/api';
  }
  
  // For Android emulator, use 10.0.2.2
  // For physical device, use your computer's local IP
  // For iOS simulator, localhost works
  if (Platform.OS === 'android') {
    // Use your local IP for physical device, or 10.0.2.2 for emulator
    return 'http://192.168.44.122:3000/api';
  }
  
  return 'http://localhost:3000/api';
};

const API_URL = getApiUrl();

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_URL,
      timeout: 10000,
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

  async getIncomeAnalysis(): Promise<ApiResponse<any>> {
    return this.client.get('/analytics/income');
  }

  // Coaching Methods
  async getFinancialAdvice(): Promise<ApiResponse<CoachingAdviceResponse>> {
    return this.client.post('/coaching/advice');
  }

  async getSpendingInsights(period: number = 30): Promise<ApiResponse<any>> {
    return this.client.get('/coaching/insights', { params: { period } });
  }

  // Health Check
  async healthCheck(): Promise<any> {
    return axios.get(`${API_URL.replace('/api', '')}/health`);
  }
}

export default new ApiService();
