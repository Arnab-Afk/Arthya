/**
 * Smart Nudge Service
 * Provides personalized financial nudges and notifications based on user behavior
 */

import { Op } from 'sequelize';
import Transaction from '../models/Transaction';
import Goal from '../models/Goal';
import Budget from '../models/Budget';
import Notification from '../models/Notification';
// Database operations handled through models

export interface Nudge {
  id: string;
  type: 'info' | 'warning' | 'success' | 'alert' | 'tip';
  category: 'spending' | 'savings' | 'income' | 'goals' | 'budget' | 'general';
  title: string;
  message: string;
  actionText?: string;
  actionType?: 'navigate' | 'dismiss' | 'action';
  actionPayload?: any;
  priority: 'low' | 'medium' | 'high' | 'urgent';
  expiresAt?: Date;
}

interface UserContext {
  userId: number;
  currentHour: number;
  dayOfWeek: number;
  dayOfMonth: number;
}

/**
 * Generate personalized nudges for a user
 */
export async function generateNudges(userId: number): Promise<Nudge[]> {
  const nudges: Nudge[] = [];
  const now = new Date();
  const context: UserContext = {
    userId,
    currentHour: now.getHours(),
    dayOfWeek: now.getDay(),
    dayOfMonth: now.getDate(),
  };

  // Get user's financial data
  const thirtyDaysAgo = new Date();
  thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

  const [transactions, goals, budgets] = await Promise.all([
    Transaction.findAll({
      where: {
        userId,
        date: { [Op.gte]: thirtyDaysAgo },
      },
    }),
    Goal.findAll({
      where: { userId, status: 'active' },
    }),
    Budget.findAll({
      where: { userId, isActive: true },
    }),
  ]);

  // Calculate basic metrics
  const income = transactions
    .filter(t => t.type === 'income')
    .reduce((sum, t) => sum + Number(t.amount), 0);

  const expenses = transactions
    .filter(t => t.type === 'expense')
    .reduce((sum, t) => sum + Number(t.amount), 0);

  const savingsRate = income > 0 ? ((income - expenses) / income) * 100 : 0;

  // Generate various nudges
  nudges.push(...generateSpendingNudges(transactions, context));
  nudges.push(...generateSavingsNudges(savingsRate, income, expenses, context));
  nudges.push(...generateGoalNudges(goals, income, context));
  nudges.push(...generateBudgetNudges(budgets, transactions, context));
  nudges.push(...generateTimeBasedNudges(context));
  nudges.push(...generateIncomeNudges(transactions, context));

  // Sort by priority and limit
  const priorityOrder = { urgent: 0, high: 1, medium: 2, low: 3 };
  nudges.sort((a, b) => priorityOrder[a.priority] - priorityOrder[b.priority]);

  // Return top 5 most relevant nudges
  return nudges.slice(0, 5);
}

/**
 * Generate spending-related nudges
 */
function generateSpendingNudges(transactions: Transaction[], context: UserContext): Nudge[] {
  const nudges: Nudge[] = [];

  const expenses = transactions.filter(t => t.type === 'expense');
  if (expenses.length === 0) return nudges;

  // Group by category
  const byCategory: { [key: string]: number } = {};
  expenses.forEach(t => {
    byCategory[t.category] = (byCategory[t.category] || 0) + Number(t.amount);
  });

  const totalExpenses = Object.values(byCategory).reduce((a, b) => a + b, 0);

  // Find dominant category
  const sortedCategories = Object.entries(byCategory)
    .sort((a, b) => b[1] - a[1]);

  if (sortedCategories.length > 0) {
    const [topCategory, topAmount] = sortedCategories[0];
    const percentage = (topAmount / totalExpenses) * 100;

    if (percentage > 40) {
      nudges.push({
        id: `spending_concentration_${topCategory}`,
        type: 'info',
        category: 'spending',
        title: 'Spending Pattern Detected',
        message: `${percentage.toFixed(0)}% of your spending is in ${topCategory}. Consider if there's room to optimize.`,
        priority: 'medium',
      });
    }
  }

  // Weekend spending alert (on Friday/Saturday)
  if (context.dayOfWeek === 5 || context.dayOfWeek === 6) {
    const weekendExpenses = expenses.filter(t => {
      const day = new Date(t.date).getDay();
      return day === 0 || day === 6;
    });

    const weekdayExpenses = expenses.filter(t => {
      const day = new Date(t.date).getDay();
      return day !== 0 && day !== 6;
    });

    const avgWeekend = weekendExpenses.length > 0
      ? weekendExpenses.reduce((s, t) => s + Number(t.amount), 0) / weekendExpenses.length
      : 0;

    const avgWeekday = weekdayExpenses.length > 0
      ? weekdayExpenses.reduce((s, t) => s + Number(t.amount), 0) / weekdayExpenses.length
      : 0;

    if (avgWeekend > avgWeekday * 1.5) {
      nudges.push({
        id: 'weekend_spending_alert',
        type: 'tip',
        category: 'spending',
        title: 'Weekend Spending Tip',
        message: `Your weekend spending is ${((avgWeekend / avgWeekday - 1) * 100).toFixed(0)}% higher than weekdays. Plan activities that are budget-friendly!`,
        priority: 'low',
      });
    }
  }

  // Recent large transaction alert
  const recentTransactions = expenses
    .filter(t => {
      const hoursDiff = (Date.now() - new Date(t.date).getTime()) / (1000 * 60 * 60);
      return hoursDiff < 24;
    })
    .sort((a, b) => Number(b.amount) - Number(a.amount));

  if (recentTransactions.length > 0) {
    const largest = recentTransactions[0];
    const avgExpense = totalExpenses / expenses.length;

    if (Number(largest.amount) > avgExpense * 3) {
      nudges.push({
        id: `large_expense_${largest.id}`,
        type: 'info',
        category: 'spending',
        title: 'Large Expense Recorded',
        message: `You spent ₹${Number(largest.amount).toLocaleString()} on ${largest.category}. This is ${(Number(largest.amount) / avgExpense).toFixed(1)}x your average expense.`,
        priority: 'medium',
      });
    }
  }

  return nudges;
}

/**
 * Generate savings-related nudges
 */
function generateSavingsNudges(
  savingsRate: number,
  income: number,
  expenses: number,
  context: UserContext
): Nudge[] {
  const nudges: Nudge[] = [];

  if (income === 0) {
    nudges.push({
      id: 'no_income_recorded',
      type: 'warning',
      category: 'income',
      title: 'No Income Recorded',
      message: 'Start tracking your income to get personalized savings recommendations.',
      actionText: 'Add Income',
      actionType: 'navigate',
      priority: 'high',
    });
    return nudges;
  }

  if (savingsRate < 10) {
    nudges.push({
      id: 'low_savings_rate',
      type: 'warning',
      category: 'savings',
      title: 'Boost Your Savings',
      message: `Your current savings rate is ${savingsRate.toFixed(0)}%. Try the 50-30-20 rule: 50% needs, 30% wants, 20% savings.`,
      priority: 'high',
    });
  } else if (savingsRate >= 20 && savingsRate < 30) {
    nudges.push({
      id: 'good_savings',
      type: 'success',
      category: 'savings',
      title: 'Great Savings Rate!',
      message: `You're saving ${savingsRate.toFixed(0)}% of your income. You're on track for financial wellness!`,
      priority: 'low',
    });
  } else if (savingsRate >= 30) {
    nudges.push({
      id: 'excellent_savings',
      type: 'success',
      category: 'savings',
      title: 'Outstanding Savings!',
      message: `${savingsRate.toFixed(0)}% savings rate is excellent! Consider investing for long-term wealth building.`,
      actionText: 'Explore Investments',
      priority: 'low',
    });
  }

  // End of month savings reminder
  if (context.dayOfMonth >= 25) {
    const potentialSavings = income - expenses;
    if (potentialSavings > 0) {
      nudges.push({
        id: 'month_end_savings',
        type: 'tip',
        category: 'savings',
        title: 'Month-End Savings',
        message: `You have ₹${potentialSavings.toLocaleString()} available. Consider transferring to savings before month-end expenses.`,
        priority: 'medium',
      });
    }
  }

  return nudges;
}

/**
 * Generate goal-related nudges
 */
function generateGoalNudges(goals: Goal[], monthlyIncome: number, context: UserContext): Nudge[] {
  const nudges: Nudge[] = [];

  if (goals.length === 0) {
    nudges.push({
      id: 'no_goals',
      type: 'tip',
      category: 'goals',
      title: 'Set Your First Goal',
      message: 'Financial goals help you stay motivated. Start with a small, achievable target!',
      actionText: 'Create Goal',
      actionType: 'navigate',
      priority: 'medium',
    });
    return nudges;
  }

  goals.forEach(goal => {
    const progress = Number(goal.currentAmount) / Number(goal.targetAmount);
    const remaining = Number(goal.targetAmount) - Number(goal.currentAmount);

    // Goal almost complete
    if (progress >= 0.9 && progress < 1) {
      nudges.push({
        id: `goal_almost_${goal.id}`,
        type: 'success',
        category: 'goals',
        title: `Almost There: ${goal.title}`,
        message: `You're ${(progress * 100).toFixed(0)}% done! Just ₹${remaining.toLocaleString()} to go!`,
        priority: 'high',
      });
    }

    // Goal deadline approaching
    if (goal.deadline) {
      const daysLeft = Math.ceil(
        (goal.deadline.getTime() - Date.now()) / (1000 * 60 * 60 * 24)
      );

      if (daysLeft <= 30 && daysLeft > 0 && progress < 0.8) {
        const monthlyNeeded = remaining / (daysLeft / 30);
        nudges.push({
          id: `goal_deadline_${goal.id}`,
          type: 'alert',
          category: 'goals',
          title: `Deadline Alert: ${goal.title}`,
          message: `${daysLeft} days left! Save ₹${monthlyNeeded.toFixed(0)}/month to reach your goal.`,
          priority: 'urgent',
        });
      }
    }

    // Suggest contribution on payday (assuming 1st or 15th)
    if ((context.dayOfMonth === 1 || context.dayOfMonth === 15) && monthlyIncome > 0) {
      const suggestedContribution = Math.min(remaining, monthlyIncome * 0.1);
      nudges.push({
        id: `goal_contribution_${goal.id}`,
        type: 'tip',
        category: 'goals',
        title: 'Payday Savings',
        message: `Consider contributing ₹${suggestedContribution.toFixed(0)} to "${goal.title}" today!`,
        actionText: 'Contribute Now',
        priority: 'medium',
      });
    }
  });

  return nudges;
}

/**
/**
 * Generate budget-related nudges
 */
function generateBudgetNudges(
  budgets: Budget[],
  transactions: Transaction[],
  _context: UserContext
): Nudge[] {
  const nudges: Nudge[] = [];

  if (budgets.length === 0) {
    nudges.push({
      id: 'create_budget',
      type: 'tip',
      category: 'budget',
      title: 'Create a Budget',
      message: 'Budgets help you control spending. Start with your top expense category!',
      actionText: 'Set Budget',
      actionType: 'navigate',
      priority: 'medium',
    });
    return nudges;
  }

  budgets.forEach(budget => {
    const categoryExpenses = transactions
      .filter(t => t.type === 'expense' && t.category === budget.category)
      .reduce((sum, t) => sum + Number(t.amount), 0);

    const percentUsed = (categoryExpenses / Number(budget.amount)) * 100;

    if (percentUsed >= 100) {
      nudges.push({
        id: `budget_exceeded_${budget.id}`,
        type: 'alert',
        category: 'budget',
        title: `Budget Exceeded: ${budget.category}`,
        message: `You've spent ₹${categoryExpenses.toLocaleString()} of ₹${Number(budget.amount).toLocaleString()} budget (${percentUsed.toFixed(0)}%).`,
        priority: 'urgent',
      });
    } else if (percentUsed >= budget.alertThreshold) {
      nudges.push({
        id: `budget_warning_${budget.id}`,
        type: 'warning',
        category: 'budget',
        title: `Budget Alert: ${budget.category}`,
        message: `${percentUsed.toFixed(0)}% of your ${budget.category} budget used. ₹${(Number(budget.amount) - categoryExpenses).toFixed(0)} remaining.`,
        priority: 'high',
      });
    }
  });

  return nudges;
}

/**
 * Generate time-based nudges
 */
function generateTimeBasedNudges(context: UserContext): Nudge[] {
  const nudges: Nudge[] = [];

  // Morning financial tip
  if (context.currentHour >= 8 && context.currentHour <= 10) {
    const tips = [
      'Review your spending from yesterday before starting your day.',
      'Pack lunch today to save on food expenses!',
      'Check your account balances to stay aware of your finances.',
      'Small daily savings add up to big results over time.',
    ];
    const tip = tips[Math.floor(Math.random() * tips.length)];

    nudges.push({
      id: 'morning_tip',
      type: 'tip',
      category: 'general',
      title: 'Morning Money Tip',
      message: tip,
      priority: 'low',
    });
  }

  // Weekly review reminder (Sunday evening)
  if (context.dayOfWeek === 0 && context.currentHour >= 18) {
    nudges.push({
      id: 'weekly_review',
      type: 'info',
      category: 'general',
      title: 'Weekly Review Time',
      message: 'Take 5 minutes to review your weekly spending and plan for next week.',
      actionText: 'View Summary',
      priority: 'medium',
    });
  }

  // Month start - set monthly goals
  if (context.dayOfMonth <= 3) {
    nudges.push({
      id: 'monthly_planning',
      type: 'tip',
      category: 'general',
      title: 'New Month, New Goals',
      message: 'Start the month right! Review last month and set this month\'s financial targets.',
      priority: 'medium',
    });
  }

  return nudges;
}

/**
 * Generate income-related nudges
 */
function generateIncomeNudges(transactions: Transaction[], _context: UserContext): Nudge[] {
  const nudges: Nudge[] = [];

  const incomeTransactions = transactions.filter(t => t.type === 'income');

  if (incomeTransactions.length === 0) return nudges;

  // Calculate income variability
  const amounts = incomeTransactions.map(t => Number(t.amount));
  const avg = amounts.reduce((a, b) => a + b, 0) / amounts.length;
  const variance = amounts.reduce((sum, a) => sum + Math.pow(a - avg, 2), 0) / amounts.length;
  const coefficientOfVariation = (Math.sqrt(variance) / avg) * 100;

  if (coefficientOfVariation > 40) {
    nudges.push({
      id: 'income_variability',
      type: 'info',
      category: 'income',
      title: 'Variable Income Detected',
      message: 'Your income varies significantly. Budget based on your lowest month and save extra during high months.',
      priority: 'medium',
    });
  }

  // Celebrate income received
  const todayIncome = incomeTransactions.filter(t => {
    const txDate = new Date(t.date).toDateString();
    return txDate === new Date().toDateString();
  });

  if (todayIncome.length > 0) {
    const totalToday = todayIncome.reduce((s, t) => s + Number(t.amount), 0);
    nudges.push({
      id: 'income_received',
      type: 'success',
      category: 'income',
      title: 'Income Received!',
      message: `₹${totalToday.toLocaleString()} received today. Remember to allocate towards savings and goals!`,
      priority: 'medium',
    });
  }

  return nudges;
}

/**
 * Save nudge as notification
 */
export async function saveNudgeAsNotification(userId: number, nudge: Nudge): Promise<void> {
  await Notification.create({
    userId,
    title: nudge.title,
    message: nudge.message,
    type: nudge.type,
    category: nudge.category,
  });
}
