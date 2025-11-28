import Transaction from '../models/Transaction';
import Goal from '../models/Goal';
import Budget from '../models/Budget';
import { Op } from 'sequelize';

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

export interface CoachingInsight {
  id: string;
  type: 'tip' | 'warning' | 'celebration' | 'action' | 'education';
  category: 'savings' | 'spending' | 'income' | 'goals' | 'budgets' | 'habits' | 'general';
  priority: 'high' | 'medium' | 'low';
  title: string;
  message: string;
  detailedExplanation?: string;
  actionItems?: string[];
  potentialSavings?: number;
  relatedGoal?: string;
}

/**
 * Build a comprehensive financial profile for a user
 */
export async function buildFinancialProfile(userId: number): Promise<FinancialProfile> {
  const threeMonthsAgo = new Date();
  threeMonthsAgo.setMonth(threeMonthsAgo.getMonth() - 3);

  const oneMonthAgo = new Date();
  oneMonthAgo.setMonth(oneMonthAgo.getMonth() - 1);

  const twoMonthsAgo = new Date();
  twoMonthsAgo.setMonth(twoMonthsAgo.getMonth() - 2);

  // Get all transactions
  const transactions = await Transaction.findAll({
    where: {
      userId,
      date: { [Op.gte]: threeMonthsAgo },
    },
    order: [['date', 'DESC']],
  });

  // Get goals and budgets
  const goals = await Goal.findAll({ where: { userId } });
  const budgets = await Budget.findAll({ where: { userId, isActive: true } });

  // Calculate income metrics
  const incomeTransactions = transactions.filter(t => t.type === 'income');
  const totalIncome = incomeTransactions.reduce((sum, t) => sum + Number(t.amount), 0);
  const avgIncome = incomeTransactions.length > 0 ? totalIncome / incomeTransactions.length : 0;

  // Income by source
  const incomeSources: { [key: string]: number } = {};
  incomeTransactions.forEach(t => {
    const cat = t.category || 'Other';
    incomeSources[cat] = (incomeSources[cat] || 0) + Number(t.amount);
  });

  // Income volatility
  let incomeVolatility = 0;
  if (incomeTransactions.length > 1) {
    const variance = incomeTransactions.reduce(
      (sum, t) => sum + Math.pow(Number(t.amount) - avgIncome, 2),
      0
    ) / incomeTransactions.length;
    incomeVolatility = Math.sqrt(variance) / avgIncome * 100;
  }

  // Calculate expense metrics
  const expenseTransactions = transactions.filter(t => t.type === 'expense');
  const totalExpenses = expenseTransactions.reduce((sum, t) => sum + Number(t.amount), 0);

  // Expenses by category
  const expensesByCategory: { [key: string]: number } = {};
  expenseTransactions.forEach(t => {
    const cat = t.category || 'Other';
    expensesByCategory[cat] = (expensesByCategory[cat] || 0) + Number(t.amount);
  });

  // Calculate trends (compare last month to previous month)
  const lastMonthIncome = transactions
    .filter(t => t.type === 'income' && t.date >= oneMonthAgo)
    .reduce((sum, t) => sum + Number(t.amount), 0);

  const prevMonthIncome = transactions
    .filter(t => t.type === 'income' && t.date >= twoMonthsAgo && t.date < oneMonthAgo)
    .reduce((sum, t) => sum + Number(t.amount), 0);

  const lastMonthExpenses = transactions
    .filter(t => t.type === 'expense' && t.date >= oneMonthAgo)
    .reduce((sum, t) => sum + Number(t.amount), 0);

  const prevMonthExpenses = transactions
    .filter(t => t.type === 'expense' && t.date >= twoMonthsAgo && t.date < oneMonthAgo)
    .reduce((sum, t) => sum + Number(t.amount), 0);

  // Determine trends
  const incomeTrend = getPercentageChange(prevMonthIncome, lastMonthIncome);
  const expenseTrend = getPercentageChange(prevMonthExpenses, lastMonthExpenses);
  const savingsTrend = getSavingsTrend(prevMonthIncome, prevMonthExpenses, lastMonthIncome, lastMonthExpenses);

  // Analyze behavior patterns
  const spendingDays = analyzeSpendingDays(expenseTransactions);
  const impulsePurchases = countImpulsePurchases(expenseTransactions);
  const recurringExpenses = findRecurringExpenses(expenseTransactions);

  // Goal analysis
  const activeGoals = goals.filter(g => g.status === 'active');
  const completedGoals = goals.filter(g => g.status === 'completed');
  const atRiskGoals = activeGoals.filter(g => isGoalAtRisk(g));

  // Budget analysis
  const budgetStatus = await analyzeBudgets(userId, budgets);

  return {
    userId,
    income: {
      total: totalIncome,
      average: avgIncome,
      sources: Object.entries(incomeSources).map(([category, amount]) => ({ category, amount })),
      volatility: incomeVolatility,
      trend: incomeTrend > 5 ? 'increasing' : incomeTrend < -5 ? 'decreasing' : 'stable',
    },
    expenses: {
      total: totalExpenses,
      average: expenseTransactions.length > 0 ? totalExpenses / expenseTransactions.length : 0,
      byCategory: Object.entries(expensesByCategory).map(([category, amount]) => ({
        category,
        amount,
        percentage: totalExpenses > 0 ? (amount / totalExpenses) * 100 : 0,
      })).sort((a, b) => b.amount - a.amount),
      trend: expenseTrend > 5 ? 'increasing' : expenseTrend < -5 ? 'decreasing' : 'stable',
    },
    savings: {
      rate: totalIncome > 0 ? ((totalIncome - totalExpenses) / totalIncome) * 100 : 0,
      netAmount: totalIncome - totalExpenses,
      trend: savingsTrend,
    },
    goals: {
      total: goals.length,
      active: activeGoals.length,
      completed: completedGoals.length,
      atRisk: atRiskGoals.length,
    },
    budgets: {
      total: budgets.length,
      onTrack: budgetStatus.onTrack,
      exceeded: budgetStatus.exceeded,
    },
    behaviorPatterns: {
      spendingDays,
      impulsePurchases,
      recurringExpenses,
    },
  };
}

/**
 * Generate personalized AI coaching insights
 */
export async function generateCoachingInsights(userId: number): Promise<CoachingInsight[]> {
  const profile = await buildFinancialProfile(userId);
  const insights: CoachingInsight[] = [];

  // 1. Savings Analysis
  if (profile.savings.rate < 0) {
    insights.push({
      id: 'negative_savings',
      type: 'warning',
      category: 'savings',
      priority: 'high',
      title: '🚨 Spending More Than Earning',
      message: `You're spending ₹${Math.abs(profile.savings.netAmount).toFixed(0)} more than you earn.`,
      detailedExplanation: 'This is unsustainable and will lead to debt. Immediate action is needed to reduce expenses or increase income.',
      actionItems: [
        'Review your top 3 expense categories for cuts',
        'Look for additional income opportunities',
        'Consider a spending freeze on non-essentials',
      ],
    });
  } else if (profile.savings.rate < 10) {
    insights.push({
      id: 'low_savings_rate',
      type: 'warning',
      category: 'savings',
      priority: 'high',
      title: 'Low Savings Rate',
      message: `Your savings rate is ${profile.savings.rate.toFixed(1)}%. Aim for at least 20%.`,
      detailedExplanation: 'Financial experts recommend saving 20% of income. For gig workers with variable income, aim even higher during good months.',
      actionItems: [
        'Set up automatic savings on payday',
        'Use the 50/30/20 rule: 50% needs, 30% wants, 20% savings',
        'Track every expense for a week to find leaks',
      ],
      potentialSavings: profile.income.total * 0.1,
    });
  } else if (profile.savings.rate >= 30) {
    insights.push({
      id: 'excellent_savings',
      type: 'celebration',
      category: 'savings',
      priority: 'low',
      title: '🎉 Excellent Saver!',
      message: `You're saving ${profile.savings.rate.toFixed(1)}% of your income. Outstanding!`,
      detailedExplanation: 'You have strong savings habits. Consider putting your money to work through investments.',
      actionItems: [
        'Research SIP (Systematic Investment Plan) options',
        'Consider opening a PPF account for tax benefits',
        'Look into index funds for long-term growth',
      ],
    });
  }

  // 2. Income Volatility (crucial for gig workers)
  if (profile.income.volatility > 40) {
    insights.push({
      id: 'high_income_volatility',
      type: 'tip',
      category: 'income',
      priority: 'high',
      title: 'Highly Variable Income',
      message: `Your income varies by ${profile.income.volatility.toFixed(0)}%. Build a larger safety net.`,
      detailedExplanation: 'Variable income makes budgeting challenging. A larger emergency fund (6+ months) is essential.',
      actionItems: [
        'Build an emergency fund covering 6 months of expenses',
        'Budget based on your lowest earning month',
        'Save extra during high-income months',
        'Consider diversifying income sources',
      ],
    });
  }

  // 3. Spending Pattern Analysis
  const topCategory = profile.expenses.byCategory[0];
  if (topCategory && topCategory.percentage > 40) {
    insights.push({
      id: 'concentrated_spending',
      type: 'warning',
      category: 'spending',
      priority: 'medium',
      title: `High ${topCategory.category} Spending`,
      message: `${topCategory.percentage.toFixed(0)}% of spending goes to ${topCategory.category}.`,
      detailedExplanation: `Spending ₹${topCategory.amount.toFixed(0)} on ${topCategory.category} is a significant portion of your budget.`,
      actionItems: [
        `Review ${topCategory.category} expenses for potential cuts`,
        'Look for alternatives or discounts',
        'Set a budget limit for this category',
      ],
      potentialSavings: topCategory.amount * 0.2,
    });
  }

  // 4. Food & Dining Analysis (common issue)
  const foodExpense = profile.expenses.byCategory.find(c => 
    c.category.toLowerCase().includes('food') || 
    c.category.toLowerCase().includes('dining') ||
    c.category.toLowerCase().includes('restaurant')
  );
  if (foodExpense && foodExpense.amount > profile.income.total * 0.15) {
    insights.push({
      id: 'high_food_spending',
      type: 'tip',
      category: 'spending',
      priority: 'medium',
      title: '🍔 Food Spending Alert',
      message: `You're spending ${((foodExpense.amount / profile.income.total) * 100).toFixed(0)}% on food.`,
      detailedExplanation: 'Food delivery apps and restaurants can quickly drain your wallet. Home cooking saves 60-70% typically.',
      actionItems: [
        'Try meal prepping on weekends',
        'Set a weekly food budget and track it',
        'Limit food delivery to once a week',
        'Pack lunch instead of ordering',
      ],
      potentialSavings: foodExpense.amount * 0.4,
    });
  }

  // 5. Weekend Spending Pattern
  if (profile.behaviorPatterns.spendingDays.includes('weekend_heavy')) {
    insights.push({
      id: 'weekend_spending',
      type: 'tip',
      category: 'habits',
      priority: 'low',
      title: 'Weekend Spending Pattern',
      message: 'You spend significantly more on weekends.',
      detailedExplanation: 'Weekend spending is often recreational and less planned. Setting a weekend budget helps.',
      actionItems: [
        'Set a fixed weekend entertainment budget',
        'Plan free activities like parks, home gatherings',
        'Wait 24 hours before weekend purchases over ₹500',
      ],
    });
  }

  // 6. Impulse Purchase Detection
  if (profile.behaviorPatterns.impulsePurchases > 5) {
    insights.push({
      id: 'impulse_purchases',
      type: 'warning',
      category: 'habits',
      priority: 'medium',
      title: 'Impulse Buying Detected',
      message: `${profile.behaviorPatterns.impulsePurchases} potential impulse purchases identified.`,
      detailedExplanation: 'Multiple small, unplanned purchases add up quickly. The "latte factor" is real.',
      actionItems: [
        'Use the 24-hour rule for non-essential purchases',
        'Unsubscribe from shopping newsletters',
        'Remove saved payment methods from apps',
        'Track every purchase for a week',
      ],
    });
  }

  // 7. Goal Progress Insights
  if (profile.goals.atRisk > 0) {
    insights.push({
      id: 'goals_at_risk',
      type: 'warning',
      category: 'goals',
      priority: 'high',
      title: `${profile.goals.atRisk} Goal(s) Need Attention`,
      message: 'Some goals may not be reached at current savings rate.',
      detailedExplanation: 'Adjusting your savings allocation or extending deadlines may be necessary.',
      actionItems: [
        'Review goal priorities and adjust if needed',
        'Increase automatic transfers to goal accounts',
        'Consider extending deadlines for less urgent goals',
      ],
    });
  }

  // 8. Budget Compliance
  if (profile.budgets.exceeded > 0) {
    insights.push({
      id: 'budgets_exceeded',
      type: 'warning',
      category: 'budgets',
      priority: 'high',
      title: `${profile.budgets.exceeded} Budget(s) Exceeded`,
      message: 'You have gone over budget in some categories.',
      detailedExplanation: 'Exceeding budgets regularly indicates they may be unrealistic or spending is out of control.',
      actionItems: [
        'Review exceeded budgets - are they realistic?',
        'Identify triggers for overspending',
        'Consider adjusting budget amounts or cutting back',
      ],
    });
  }

  // 9. No Budgets Set
  if (profile.budgets.total === 0) {
    insights.push({
      id: 'no_budgets',
      type: 'action',
      category: 'budgets',
      priority: 'medium',
      title: 'Set Up Your First Budget',
      message: 'Budgets help you control spending and reach goals faster.',
      detailedExplanation: 'People with budgets save 20% more on average. Start with your highest expense category.',
      actionItems: [
        `Create a budget for ${topCategory?.category || 'your top expense category'}`,
        'Set realistic limits based on past spending',
        'Review and adjust monthly',
      ],
    });
  }

  // 10. No Goals Set
  if (profile.goals.total === 0) {
    insights.push({
      id: 'no_goals',
      type: 'action',
      category: 'goals',
      priority: 'medium',
      title: 'Set Your First Financial Goal',
      message: 'Goals give purpose to your savings.',
      detailedExplanation: 'Having specific goals makes you 42% more likely to achieve them.',
      actionItems: [
        'Start with an emergency fund goal (3-6 months expenses)',
        'Add a fun goal to stay motivated',
        'Break big goals into smaller milestones',
      ],
    });
  }

  // 11. Recurring Expenses Review
  if (profile.behaviorPatterns.recurringExpenses.length > 3) {
    const monthlyRecurring = profile.behaviorPatterns.recurringExpenses
      .reduce((sum, e) => sum + e.amount, 0);
    
    insights.push({
      id: 'review_subscriptions',
      type: 'tip',
      category: 'spending',
      priority: 'low',
      title: 'Review Recurring Expenses',
      message: `You have ${profile.behaviorPatterns.recurringExpenses.length} recurring expenses totaling ₹${monthlyRecurring}/month.`,
      detailedExplanation: 'Subscription creep is common. Review each recurring expense to ensure you still use it.',
      actionItems: [
        'List all subscriptions and their usage',
        'Cancel unused subscriptions',
        'Look for annual payment discounts',
        'Share family plans where possible',
      ],
      potentialSavings: monthlyRecurring * 0.2,
    });
  }

  // 12. Positive Trend Recognition
  if (profile.savings.trend === 'improving') {
    insights.push({
      id: 'improving_savings',
      type: 'celebration',
      category: 'savings',
      priority: 'low',
      title: '📈 Your Savings Are Improving!',
      message: 'Great job! Your savings rate is trending upward.',
      actionItems: [
        'Keep up the momentum',
        'Consider increasing your savings goal',
        'Celebrate milestones (responsibly!)',
      ],
    });
  }

  // 13. Education Tips (rotate these)
  const educationTips = [
    {
      id: 'education_emergency_fund',
      title: '💡 Emergency Fund Basics',
      message: 'An emergency fund should cover 3-6 months of expenses.',
      detailedExplanation: 'This money should be easily accessible (savings account) but separate from daily spending.',
    },
    {
      id: 'education_compound_interest',
      title: '💡 Power of Compound Interest',
      message: 'Starting to invest early makes a huge difference.',
      detailedExplanation: 'Investing ₹5,000/month from age 25 vs 35 could mean ₹1 crore more by retirement.',
    },
    {
      id: 'education_50_30_20',
      title: '💡 The 50/30/20 Rule',
      message: 'A simple budgeting framework: 50% needs, 30% wants, 20% savings.',
      detailedExplanation: 'This rule provides a balanced approach to managing money without being too restrictive.',
    },
  ];

  // Add a random education tip
  const randomTip = educationTips[Math.floor(Math.random() * educationTips.length)];
  insights.push({
    ...randomTip,
    type: 'education',
    category: 'general',
    priority: 'low',
  });

  // Sort by priority
  const priorityOrder = { high: 0, medium: 1, low: 2 };
  insights.sort((a, b) => priorityOrder[a.priority] - priorityOrder[b.priority]);

  return insights;
}

/**
 * Generate a weekly financial summary
 */
export async function generateWeeklySummary(userId: number): Promise<{
  period: string;
  income: number;
  expenses: number;
  savings: number;
  topCategories: { category: string; amount: number }[];
  comparison: { incomeChange: number; expenseChange: number };
  highlights: string[];
}> {
  const oneWeekAgo = new Date();
  oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);

  const twoWeeksAgo = new Date();
  twoWeeksAgo.setDate(twoWeeksAgo.getDate() - 14);

  const thisWeekTxns = await Transaction.findAll({
    where: {
      userId,
      date: { [Op.gte]: oneWeekAgo },
    },
  });

  const lastWeekTxns = await Transaction.findAll({
    where: {
      userId,
      date: { [Op.gte]: twoWeeksAgo, [Op.lt]: oneWeekAgo },
    },
  });

  const thisWeekIncome = thisWeekTxns
    .filter(t => t.type === 'income')
    .reduce((sum, t) => sum + Number(t.amount), 0);

  const thisWeekExpenses = thisWeekTxns
    .filter(t => t.type === 'expense')
    .reduce((sum, t) => sum + Number(t.amount), 0);

  const lastWeekIncome = lastWeekTxns
    .filter(t => t.type === 'income')
    .reduce((sum, t) => sum + Number(t.amount), 0);

  const lastWeekExpenses = lastWeekTxns
    .filter(t => t.type === 'expense')
    .reduce((sum, t) => sum + Number(t.amount), 0);

  // Top categories
  const categoryTotals: { [key: string]: number } = {};
  thisWeekTxns
    .filter(t => t.type === 'expense')
    .forEach(t => {
      categoryTotals[t.category] = (categoryTotals[t.category] || 0) + Number(t.amount);
    });

  const topCategories = Object.entries(categoryTotals)
    .map(([category, amount]) => ({ category, amount }))
    .sort((a, b) => b.amount - a.amount)
    .slice(0, 5);

  // Generate highlights
  const highlights: string[] = [];
  
  const incomeChange = lastWeekIncome > 0 
    ? ((thisWeekIncome - lastWeekIncome) / lastWeekIncome) * 100 
    : 0;
  
  const expenseChange = lastWeekExpenses > 0 
    ? ((thisWeekExpenses - lastWeekExpenses) / lastWeekExpenses) * 100 
    : 0;

  if (incomeChange > 10) {
    highlights.push(`💰 Income up ${incomeChange.toFixed(0)}% from last week!`);
  }
  if (expenseChange < -10) {
    highlights.push(`✅ Expenses down ${Math.abs(expenseChange).toFixed(0)}% - great job!`);
  }
  if (thisWeekIncome > thisWeekExpenses) {
    highlights.push(`💚 Positive cash flow: ₹${(thisWeekIncome - thisWeekExpenses).toFixed(0)} saved`);
  }

  return {
    period: `${oneWeekAgo.toLocaleDateString()} - ${new Date().toLocaleDateString()}`,
    income: thisWeekIncome,
    expenses: thisWeekExpenses,
    savings: thisWeekIncome - thisWeekExpenses,
    topCategories,
    comparison: {
      incomeChange,
      expenseChange,
    },
    highlights,
  };
}

// Helper functions
function getPercentageChange(previous: number, current: number): number {
  if (previous === 0) return 0;
  return ((current - previous) / previous) * 100;
}

function getSavingsTrend(
  prevIncome: number, 
  prevExpense: number, 
  currIncome: number, 
  currExpense: number
): 'improving' | 'declining' | 'stable' {
  const prevRate = prevIncome > 0 ? (prevIncome - prevExpense) / prevIncome : 0;
  const currRate = currIncome > 0 ? (currIncome - currExpense) / currIncome : 0;
  
  if (currRate > prevRate + 0.05) return 'improving';
  if (currRate < prevRate - 0.05) return 'declining';
  return 'stable';
}

function analyzeSpendingDays(expenses: Transaction[]): string[] {
  const patterns: string[] = [];
  
  const dayTotals: { [key: number]: number } = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0 };
  expenses.forEach(t => {
    const day = t.date.getDay();
    dayTotals[day] += Number(t.amount);
  });

  const weekdayTotal = dayTotals[1] + dayTotals[2] + dayTotals[3] + dayTotals[4] + dayTotals[5];
  const weekendTotal = dayTotals[0] + dayTotals[6];

  if (weekendTotal > weekdayTotal * 0.6) {
    patterns.push('weekend_heavy');
  }

  return patterns;
}

function countImpulsePurchases(expenses: Transaction[]): number {
  // Consider impulse: multiple small purchases on same day, unusual timing, etc.
  const purchasesByDay: { [key: string]: Transaction[] } = {};
  
  expenses.forEach(t => {
    const day = t.date.toISOString().split('T')[0];
    if (!purchasesByDay[day]) purchasesByDay[day] = [];
    purchasesByDay[day].push(t);
  });

  let impulseCount = 0;
  Object.values(purchasesByDay).forEach(dayTxns => {
    // More than 3 small purchases in a day might indicate impulse buying
    const smallPurchases = dayTxns.filter(t => Number(t.amount) < 500);
    if (smallPurchases.length >= 3) {
      impulseCount += smallPurchases.length - 2;
    }
  });

  return impulseCount;
}

function findRecurringExpenses(expenses: Transaction[]): { description: string; amount: number; frequency: string }[] {
  const descriptionCounts: { [key: string]: { count: number; amount: number } } = {};
  
  expenses.forEach(t => {
    const desc = t.description?.toLowerCase() || '';
    if (desc.length > 3) {
      if (!descriptionCounts[desc]) {
        descriptionCounts[desc] = { count: 0, amount: 0 };
      }
      descriptionCounts[desc].count++;
      descriptionCounts[desc].amount = Number(t.amount);
    }
  });

  return Object.entries(descriptionCounts)
    .filter(([_, data]) => data.count >= 2)
    .map(([description, data]) => ({
      description,
      amount: data.amount,
      frequency: data.count >= 4 ? 'weekly' : 'monthly',
    }))
    .slice(0, 10);
}

function isGoalAtRisk(goal: Goal): boolean {
  if (!goal.deadline || goal.status !== 'active') return false;
  
  const now = new Date();
  const deadline = new Date(goal.deadline);
  const daysLeft = (deadline.getTime() - now.getTime()) / (1000 * 60 * 60 * 24);
  
  if (daysLeft <= 0) return true;
  
  const progress = Number(goal.currentAmount) / Number(goal.targetAmount);
  const timeProgress = 1 - (daysLeft / 365); // Assume 1 year goals
  
  return progress < timeProgress * 0.7;
}

async function analyzeBudgets(userId: number, budgets: Budget[]): Promise<{ onTrack: number; exceeded: number }> {
  let onTrack = 0;
  let exceeded = 0;

  for (const budget of budgets) {
    const spent = await Transaction.sum('amount', {
      where: {
        userId,
        type: 'expense',
        category: budget.category,
        date: {
          [Op.gte]: budget.startDate,
          [Op.lte]: budget.endDate || new Date(),
        },
      },
    }) || 0;

    if (spent > Number(budget.amount)) {
      exceeded++;
    } else {
      onTrack++;
    }
  }

  return { onTrack, exceeded };
}
