import { Response } from 'express';
import { Op } from 'sequelize';
import Transaction from '../models/Transaction';
import Goal from '../models/Goal';
import Notification from '../models/Notification';
import { AuthRequest } from '../middleware/auth';
import { 
  buildFinancialProfile, 
  generateCoachingInsights, 
  generateWeeklySummary
} from '../services/coachingAI';
import {
  generateAIAdvice,
  chatWithAI,
  generateAIWeeklySummary,
  analyzeSpendingPatterns,
  generateGoalAdvice
} from '../services/geminiService';

// @desc    Get AI financial coaching advice
// @route   POST /api/coaching/advice
// @access  Private
export const getFinancialAdvice = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const userId = req.user?.id;

    // Get recent financial data
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

    const recentTransactions = await Transaction.findAll({
      where: {
        userId,
        date: { [Op.gte]: thirtyDaysAgo },
      },
    });

    const income = recentTransactions
      .filter((t) => t.type === 'income')
      .reduce((sum, t) => sum + Number(t.amount), 0);

    const expenses = recentTransactions
      .filter((t) => t.type === 'expense')
      .reduce((sum, t) => sum + Number(t.amount), 0);

    const savingsRate = income > 0 ? ((income - expenses) / income) * 100 : 0;

    // Get active goals
    const activeGoals = await Goal.findAll({
      where: { userId, status: 'active' },
    });

    // Generate personalized advice
    const advice = [];

    // Savings rate advice
    if (savingsRate < 10) {
      advice.push({
        type: 'warning',
        category: 'savings',
        title: 'Low Savings Rate',
        message: `Your savings rate is ${savingsRate.toFixed(1)}%. Try to save at least 20% of your income for a healthier financial future.`,
        actionable: 'Set up automatic transfers to savings after each paycheck.',
      });
    } else if (savingsRate >= 20) {
      advice.push({
        type: 'success',
        category: 'savings',
        title: 'Excellent Savings!',
        message: `You're saving ${savingsRate.toFixed(1)}% of your income. Keep up the great work!`,
        actionable: 'Consider investing some of your savings for long-term growth.',
      });
    }

    // Expense analysis
    const expensesByCategory = recentTransactions
      .filter((t) => t.type === 'expense')
      .reduce((acc: any, t) => {
        acc[t.category] = (acc[t.category] || 0) + Number(t.amount);
        return acc;
      }, {});

    const topExpenseCategory = Object.entries(expensesByCategory).sort(
      (a: any, b: any) => b[1] - a[1]
    )[0];

    if (topExpenseCategory) {
      const percentage = ((topExpenseCategory[1] as number) / expenses) * 100;
      if (percentage > 30) {
        advice.push({
          type: 'info',
          category: 'spending',
          title: 'High Spending in One Category',
          message: `${percentage.toFixed(1)}% of your expenses are in ${topExpenseCategory[0]}. Consider reviewing these expenses.`,
          actionable: `Look for ways to reduce ${topExpenseCategory[0]} costs.`,
        });
      }
    }

    // Goal progress advice
    activeGoals.forEach((goal) => {
      const progress = (Number(goal.currentAmount) / Number(goal.targetAmount)) * 100;
      if (progress < 50 && goal.deadline) {
        const daysLeft = Math.ceil(
          (goal.deadline.getTime() - Date.now()) / (1000 * 60 * 60 * 24)
        );
        if (daysLeft < 90) {
          const monthlyNeeded =
            (Number(goal.targetAmount) - Number(goal.currentAmount)) / (daysLeft / 30);
          advice.push({
            type: 'alert',
            category: 'goals',
            title: `Goal Deadline Approaching: ${goal.title}`,
            message: `You need to save approximately $${monthlyNeeded.toFixed(2)} per month to reach this goal on time.`,
            actionable: 'Adjust your budget to allocate more towards this goal.',
          });
        }
      }
    });

    // Income variability advice (for gig workers)
    const incomeTransactions = recentTransactions.filter(
      (t) => t.type === 'income'
    );
    if (incomeTransactions.length > 0) {
      const avgIncome = income / incomeTransactions.length;
      const variance = incomeTransactions.reduce(
        (sum, t) => sum + Math.pow(Number(t.amount) - avgIncome, 2),
        0
      );
      const stdDev = Math.sqrt(variance / incomeTransactions.length);
      const coefficientOfVariation = (stdDev / avgIncome) * 100;

      if (coefficientOfVariation > 30) {
        advice.push({
          type: 'info',
          category: 'income',
          title: 'Variable Income Detected',
          message: 'Your income varies significantly. Build an emergency fund covering 3-6 months of expenses.',
          actionable: 'Save extra during high-income months to cover low-income periods.',
        });
      }
    }

    res.status(200).json({
      success: true,
      data: {
        advice,
        summary: {
          income: Number(income.toFixed(2)),
          expenses: Number(expenses.toFixed(2)),
          savingsRate: Number(savingsRate.toFixed(1)),
          activeGoals: activeGoals.length,
        },
      },
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Get spending insights
// @route   GET /api/coaching/insights
// @access  Private
export const getSpendingInsights = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const userId = req.user?.id;
    const { period = '30' } = req.query;

    const daysAgo = new Date();
    daysAgo.setDate(daysAgo.getDate() - Number(period));

    const transactions = await Transaction.findAll({
      where: {
        userId,
        type: 'expense',
        date: { [Op.gte]: daysAgo },
      },
    });

    const insights = {
      totalSpent: 0,
      averageTransaction: 0,
      mostExpensiveDay: '',
      mostExpensiveCategory: '',
      unusualSpending: [],
    };

    if (transactions.length > 0) {
      insights.totalSpent = transactions.reduce(
        (sum, t) => sum + Number(t.amount),
        0
      );
      insights.averageTransaction = insights.totalSpent / transactions.length;

      // Group by day
      const spendingByDay: any = {};
      transactions.forEach((t) => {
        const day = t.date.toISOString().split('T')[0];
        spendingByDay[day] = (spendingByDay[day] || 0) + Number(t.amount);
      });

      insights.mostExpensiveDay = Object.entries(spendingByDay).sort(
        (a: any, b: any) => b[1] - a[1]
      )[0]?.[0] || '';

      // Group by category
      const spendingByCategory: any = {};
      transactions.forEach((t) => {
        spendingByCategory[t.category] =
          (spendingByCategory[t.category] || 0) + Number(t.amount);
      });

      insights.mostExpensiveCategory = Object.entries(spendingByCategory).sort(
        (a: any, b: any) => b[1] - a[1]
      )[0]?.[0] || '';
    }

    res.status(200).json({
      success: true,
      data: insights,
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Create smart notification
// @route   POST /api/coaching/notify
// @access  Private
export const createSmartNotification = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const { title, message, type, category } = req.body;

    const notification = await Notification.create({
      userId: req.user?.id,
      title,
      message,
      type,
      category,
    });

    res.status(201).json({
      success: true,
      data: notification,
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Get AI-powered coaching insights
// @route   GET /api/coaching/ai-insights
// @access  Private
export const getAIInsights = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const userId = req.user?.id;
    const insights = await generateCoachingInsights(userId!);

    res.status(200).json({
      success: true,
      data: {
        insights,
        generatedAt: new Date().toISOString(),
        totalInsights: insights.length,
        highPriority: insights.filter(i => i.priority === 'high').length,
      },
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Get financial profile
// @route   GET /api/coaching/profile
// @access  Private
export const getFinancialProfile = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const userId = req.user?.id;
    const profile = await buildFinancialProfile(userId!);

    res.status(200).json({
      success: true,
      data: profile,
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Get weekly summary
// @route   GET /api/coaching/weekly-summary
// @access  Private
export const getWeeklySummary = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const userId = req.user?.id;
    const summary = await generateWeeklySummary(userId!);

    res.status(200).json({
      success: true,
      data: summary,
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Get personalized action plan
// @route   GET /api/coaching/action-plan
// @access  Private
export const getActionPlan = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const userId = req.user?.id;
    const profile = await buildFinancialProfile(userId!);
    const insights = await generateCoachingInsights(userId!);

    // Generate a prioritized action plan based on profile and insights
    const actionPlan = {
      immediateActions: [] as { action: string; reason: string; impact: string }[],
      weeklyGoals: [] as { goal: string; target: string }[],
      monthlyTargets: [] as { target: string; metric: string; current: string }[],
      learningResources: [] as { topic: string; reason: string }[],
    };

    // Immediate actions based on high priority insights
    const highPriorityInsights = insights.filter(i => i.priority === 'high');
    highPriorityInsights.forEach(insight => {
      if (insight.actionItems && insight.actionItems.length > 0) {
        actionPlan.immediateActions.push({
          action: insight.actionItems[0],
          reason: insight.message,
          impact: insight.potentialSavings 
            ? `Could save ₹${insight.potentialSavings.toFixed(0)}/month` 
            : 'Improve financial health',
        });
      }
    });

    // Weekly goals
    if (profile.savings.rate < 20) {
      actionPlan.weeklyGoals.push({
        goal: 'Track all expenses this week',
        target: 'Log every purchase, no matter how small',
      });
    }

    if (profile.expenses.byCategory.length > 0) {
      actionPlan.weeklyGoals.push({
        goal: `Reduce ${profile.expenses.byCategory[0].category} spending by 10%`,
        target: `Keep under ₹${(profile.expenses.byCategory[0].amount * 0.9 / 4).toFixed(0)} this week`,
      });
    }

    // Monthly targets
    actionPlan.monthlyTargets.push({
      target: 'Savings Rate',
      metric: '20%',
      current: `${profile.savings.rate.toFixed(1)}%`,
    });

    if (profile.goals.active > 0) {
      actionPlan.monthlyTargets.push({
        target: 'Goal Progress',
        metric: 'On track',
        current: profile.goals.atRisk > 0 ? `${profile.goals.atRisk} at risk` : 'All on track',
      });
    }

    // Learning resources based on needs
    if (profile.income.volatility > 30) {
      actionPlan.learningResources.push({
        topic: 'Managing Variable Income',
        reason: 'Your income varies significantly',
      });
    }

    if (profile.savings.rate >= 20) {
      actionPlan.learningResources.push({
        topic: 'Introduction to Investing',
        reason: 'You have a good savings rate - time to grow your money',
      });
    }

    res.status(200).json({
      success: true,
      data: actionPlan,
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Chat with AI coach
// @route   POST /api/coaching/chat
// @access  Private
export const chatWithCoach = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const userId = req.user?.id;
    const { message, history = [] } = req.body;

    if (!message) {
      res.status(400).json({
        success: false,
        message: 'Message is required',
      });
      return;
    }

    // Get user context for personalized responses
    const profile = await buildFinancialProfile(userId!);
    
    // Use Gemini AI for chat
    const aiResponse = await chatWithAI(message, profile, history);

    res.status(200).json({
      success: true,
      data: {
        message: aiResponse.response,
        suggestions: aiResponse.suggestions,
        context: {
          savingsRate: profile.savings.rate.toFixed(1),
          topCategory: profile.expenses.byCategory[0]?.category,
          activeGoals: profile.goals.active,
        },
      },
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Get AI-powered comprehensive advice (Gemini)
// @route   GET /api/coaching/gemini-advice
// @access  Private
export const getGeminiAdvice = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const userId = req.user?.id;
    const profile = await buildFinancialProfile(userId!);
    
    const advice = await generateAIAdvice(profile);

    res.status(200).json({
      success: true,
      data: {
        ...advice,
        generatedAt: new Date().toISOString(),
        model: 'gemini-1.5-flash',
      },
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Get AI weekly summary (Gemini)
// @route   GET /api/coaching/gemini-weekly
// @access  Private
export const getGeminiWeeklySummary = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const userId = req.user?.id;
    const profile = await buildFinancialProfile(userId!);
    
    // Get this week's data
    const oneWeekAgo = new Date();
    oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);
    const twoWeeksAgo = new Date();
    twoWeeksAgo.setDate(twoWeeksAgo.getDate() - 14);

    const thisWeekTransactions = await Transaction.findAll({
      where: {
        userId,
        date: { [Op.gte]: oneWeekAgo },
      },
    });

    const lastWeekTransactions = await Transaction.findAll({
      where: {
        userId,
        date: { [Op.gte]: twoWeeksAgo, [Op.lt]: oneWeekAgo },
      },
    });

    const thisWeekIncome = thisWeekTransactions
      .filter(t => t.type === 'income')
      .reduce((sum, t) => sum + Number(t.amount), 0);
    const thisWeekExpenses = thisWeekTransactions
      .filter(t => t.type === 'expense')
      .reduce((sum, t) => sum + Number(t.amount), 0);

    const lastWeekIncome = lastWeekTransactions
      .filter(t => t.type === 'income')
      .reduce((sum, t) => sum + Number(t.amount), 0);
    const lastWeekExpenses = lastWeekTransactions
      .filter(t => t.type === 'expense')
      .reduce((sum, t) => sum + Number(t.amount), 0);

    // Get top categories
    const categorySpending: { [key: string]: number } = {};
    thisWeekTransactions
      .filter(t => t.type === 'expense')
      .forEach(t => {
        categorySpending[t.category] = (categorySpending[t.category] || 0) + Number(t.amount);
      });

    const topCategories = Object.entries(categorySpending)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([category, amount]) => ({ category, amount }));

    const weeklyData = {
      income: thisWeekIncome,
      expenses: thisWeekExpenses,
      savings: thisWeekIncome - thisWeekExpenses,
      topCategories,
      comparedToLastWeek: {
        incomeChange: lastWeekIncome > 0 
          ? ((thisWeekIncome - lastWeekIncome) / lastWeekIncome) * 100 
          : 0,
        expenseChange: lastWeekExpenses > 0 
          ? ((thisWeekExpenses - lastWeekExpenses) / lastWeekExpenses) * 100 
          : 0,
      },
    };

    const summary = await generateAIWeeklySummary(weeklyData, profile);

    res.status(200).json({
      success: true,
      data: {
        ...summary,
        weeklyData,
        generatedAt: new Date().toISOString(),
        model: 'gemini-1.5-flash',
      },
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Get spending pattern analysis (Gemini)
// @route   GET /api/coaching/spending-analysis
// @access  Private
export const getSpendingAnalysis = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const userId = req.user?.id;
    const { days = 30 } = req.query;

    const daysAgo = new Date();
    daysAgo.setDate(daysAgo.getDate() - Number(days));

    const transactions = await Transaction.findAll({
      where: {
        userId,
        type: 'expense',
        date: { [Op.gte]: daysAgo },
      },
    });

    const profile = await buildFinancialProfile(userId!);

    const transactionData = transactions.map(t => ({
      category: t.category,
      amount: Number(t.amount),
      date: t.date,
      description: t.description || '',
    }));

    const analysis = await analyzeSpendingPatterns(transactionData, profile);

    res.status(200).json({
      success: true,
      data: {
        ...analysis,
        period: `Last ${days} days`,
        transactionCount: transactions.length,
        generatedAt: new Date().toISOString(),
        model: 'gemini-1.5-flash',
      },
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Get AI goal advice (Gemini)
// @route   GET /api/coaching/goal-advice/:goalId
// @access  Private
export const getGoalAdvice = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const userId = req.user?.id;
    const { goalId } = req.params;

    const goal = await Goal.findOne({
      where: { id: goalId, userId },
    });

    if (!goal) {
      res.status(404).json({
        success: false,
        message: 'Goal not found',
      });
      return;
    }

    const profile = await buildFinancialProfile(userId!);

    const advice = await generateGoalAdvice(
      {
        title: goal.title,
        targetAmount: Number(goal.targetAmount),
        currentAmount: Number(goal.currentAmount),
        deadline: goal.deadline || undefined,
      },
      profile
    );

    res.status(200).json({
      success: true,
      data: {
        goal: {
          id: goal.id,
          title: goal.title,
          progress: ((Number(goal.currentAmount) / Number(goal.targetAmount)) * 100).toFixed(1),
        },
        ...advice,
        generatedAt: new Date().toISOString(),
        model: 'gemini-1.5-flash',
      },
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};
