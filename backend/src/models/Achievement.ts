import { Table, Column, Model, DataType, ForeignKey, BelongsTo } from 'sequelize-typescript';
import User from './User';

@Table({
  tableName: 'achievements',
  timestamps: true,
})
export default class Achievement extends Model {
  @ForeignKey(() => User)
  @Column({
    type: DataType.INTEGER,
    allowNull: false,
  })
  userId!: number;

  @BelongsTo(() => User)
  user!: User;

  @Column({
    type: DataType.STRING,
    allowNull: false,
  })
  type!: string;

  @Column({
    type: DataType.STRING,
    allowNull: false,
  })
  title!: string;

  @Column({
    type: DataType.TEXT,
    allowNull: true,
  })
  description?: string;

  @Column({
    type: DataType.STRING,
    allowNull: true,
  })
  icon?: string;

  @Column({
    type: DataType.STRING,
    allowNull: true,
  })
  badge?: string;

  @Column({
    type: DataType.INTEGER,
    defaultValue: 0,
  })
  points!: number;

  @Column({
    type: DataType.DATE,
    defaultValue: DataType.NOW,
  })
  unlockedAt!: Date;

  @Column({
    type: DataType.JSONB,
    allowNull: true,
  })
  metadata?: {
    targetValue?: number;
    achievedValue?: number;
    streak?: number;
    category?: string;
  };
}

// Achievement types and definitions
export const ACHIEVEMENT_DEFINITIONS = {
  // Savings achievements
  first_savings: {
    title: 'First Step',
    description: 'Made your first savings contribution',
    icon: '🌱',
    points: 10,
  },
  savings_streak_7: {
    title: 'Week Warrior',
    description: 'Saved money for 7 consecutive days',
    icon: '🔥',
    points: 50,
  },
  savings_streak_30: {
    title: 'Monthly Master',
    description: 'Saved money for 30 consecutive days',
    icon: '💪',
    points: 200,
  },
  emergency_fund_1month: {
    title: 'Safety Net',
    description: 'Built 1 month of emergency fund',
    icon: '🛡️',
    points: 100,
  },
  emergency_fund_3months: {
    title: 'Financially Secure',
    description: 'Built 3 months of emergency fund',
    icon: '🏰',
    points: 300,
  },
  emergency_fund_6months: {
    title: 'Fort Knox',
    description: 'Built 6 months of emergency fund',
    icon: '🏆',
    points: 500,
  },

  // Goal achievements
  first_goal: {
    title: 'Goal Setter',
    description: 'Created your first financial goal',
    icon: '🎯',
    points: 10,
  },
  goal_completed: {
    title: 'Goal Crusher',
    description: 'Completed a financial goal',
    icon: '✅',
    points: 100,
  },
  goals_5_completed: {
    title: 'Achievement Hunter',
    description: 'Completed 5 financial goals',
    icon: '🏅',
    points: 250,
  },

  // Budget achievements
  budget_created: {
    title: 'Budget Beginner',
    description: 'Created your first budget',
    icon: '📊',
    points: 10,
  },
  budget_under_7days: {
    title: 'Budget Keeper',
    description: 'Stayed under budget for 7 days',
    icon: '💰',
    points: 50,
  },
  budget_under_30days: {
    title: 'Budget Master',
    description: 'Stayed under budget for a full month',
    icon: '👑',
    points: 200,
  },

  // Income achievements
  income_growth_10: {
    title: 'Growing Strong',
    description: 'Increased income by 10% compared to last month',
    icon: '📈',
    points: 75,
  },
  income_growth_25: {
    title: 'Skyrocketing',
    description: 'Increased income by 25% compared to last month',
    icon: '🚀',
    points: 150,
  },
  diversified_income: {
    title: 'Income Diversifier',
    description: 'Earned from 3+ different sources',
    icon: '🌈',
    points: 100,
  },

  // Expense tracking
  tracker_7days: {
    title: 'Tracking Rookie',
    description: 'Tracked expenses for 7 consecutive days',
    icon: '📝',
    points: 25,
  },
  tracker_30days: {
    title: 'Tracking Pro',
    description: 'Tracked expenses for 30 consecutive days',
    icon: '📋',
    points: 100,
  },
  spending_reduced_10: {
    title: 'Thrifty',
    description: 'Reduced spending by 10% compared to last month',
    icon: '💵',
    points: 75,
  },
  spending_reduced_25: {
    title: 'Super Saver',
    description: 'Reduced spending by 25% compared to last month',
    icon: '🦸',
    points: 150,
  },

  // Special achievements
  savings_rate_20: {
    title: 'Savings Champion',
    description: 'Achieved 20% savings rate',
    icon: '🏆',
    points: 100,
  },
  savings_rate_50: {
    title: 'Financial Freedom Seeker',
    description: 'Achieved 50% savings rate',
    icon: '🔥',
    points: 300,
  },
  debt_free: {
    title: 'Debt Free',
    description: 'Paid off all debts',
    icon: '🎉',
    points: 500,
  },
  first_investment: {
    title: 'Investor',
    description: 'Made your first investment',
    icon: '📊',
    points: 50,
  },

  // Engagement achievements
  coach_used_7days: {
    title: 'Advice Seeker',
    description: 'Used financial coach for 7 days',
    icon: '🎓',
    points: 25,
  },
  profile_complete: {
    title: 'All Set',
    description: 'Completed your profile',
    icon: '✨',
    points: 15,
  },
  sms_imported: {
    title: 'Auto Tracker',
    description: 'Imported transactions from SMS',
    icon: '📱',
    points: 20,
  },
};
