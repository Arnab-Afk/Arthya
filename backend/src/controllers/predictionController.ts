import { Response } from 'express';
import { AuthRequest } from '../middleware/auth';
import { predictIncome } from '../services/incomePrediction';
import { generateNudges } from '../services/nudgeService';
import Transaction from '../models/Transaction';
import { Op } from 'sequelize';

// @desc    Get income prediction
// @route   GET /api/predictions/income
// @access  Private
export const getIncomePrediction = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const userId = req.user?.id;
    const months = parseInt(req.query.months as string) || 3;

    const prediction = await predictIncome(userId!, months);

    res.status(200).json({
      success: true,
      data: prediction,
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Get expense prediction
// @route   GET /api/predictions/expenses
// @access  Private
export const getExpensePrediction = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const userId = req.user?.id;
    const months = parseInt(req.query.months as string) || 3;

    // Get expense history
    const sixMonthsAgo = new Date();
    sixMonthsAgo.setMonth(sixMonthsAgo.getMonth() - 6);

    const expenses = await Transaction.findAll({
      where: {
        userId,
        type: 'expense',
        date: { [Op.gte]: sixMonthsAgo },
      },
      order: [['date', 'ASC']],
    });

    // Group by month
    const monthlyExpenses: { [key: string]: number } = {};
    expenses.forEach((t) => {
      const monthKey = `${t.date.getFullYear()}-${String(t.date.getMonth() + 1).padStart(2, '0')}`;
      monthlyExpenses[monthKey] = (monthlyExpenses[monthKey] || 0) + Number(t.amount);
    });

    const values = Object.values(monthlyExpenses);
    if (values.length < 2) {
      res.status(200).json({
        success: true,
        data: {
          predictions: [],
          message: 'Not enough data for expense prediction',
        },
      });
      return;
    }

    // Simple moving average prediction
    const avgExpense = values.reduce((a, b) => a + b, 0) / values.length;
    const predictions: { month: string; predicted: number }[] = [];

    const currentDate = new Date();
    for (let i = 1; i <= months; i++) {
      const futureDate = new Date(currentDate);
      futureDate.setMonth(futureDate.getMonth() + i);
      predictions.push({
        month: `${futureDate.getFullYear()}-${String(futureDate.getMonth() + 1).padStart(2, '0')}`,
        predicted: Math.round(avgExpense * (1 + (Math.random() * 0.1 - 0.05))), // Add slight variance
      });
    }

    // Category breakdown prediction
    const categoryExpenses: { [key: string]: number[] } = {};
    expenses.forEach((t) => {
      const cat = t.category || 'Other';
      if (!categoryExpenses[cat]) categoryExpenses[cat] = [];
      categoryExpenses[cat].push(Number(t.amount));
    });

    const categoryPredictions = Object.entries(categoryExpenses).map(([category, amounts]) => ({
      category,
      avgMonthly: Math.round(amounts.reduce((a, b) => a + b, 0) / values.length),
      trend: amounts.length > 1 ? (amounts[amounts.length - 1] > amounts[0] ? 'increasing' : 'decreasing') : 'stable',
    }));

    res.status(200).json({
      success: true,
      data: {
        predictions,
        categoryPredictions,
        historicalAverage: Math.round(avgExpense),
        dataPointsUsed: values.length,
      },
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Get cash flow projection
// @route   GET /api/predictions/cashflow
// @access  Private
export const getCashFlowProjection = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const userId = req.user?.id;
    const months = parseInt(req.query.months as string) || 3;

    // Get historical data
    const threeMonthsAgo = new Date();
    threeMonthsAgo.setMonth(threeMonthsAgo.getMonth() - 3);

    const transactions = await Transaction.findAll({
      where: {
        userId,
        date: { [Op.gte]: threeMonthsAgo },
      },
      order: [['date', 'ASC']],
    });

    // Calculate averages
    const monthlyData: { [key: string]: { income: number; expense: number } } = {};
    transactions.forEach((t) => {
      const monthKey = `${t.date.getFullYear()}-${String(t.date.getMonth() + 1).padStart(2, '0')}`;
      if (!monthlyData[monthKey]) {
        monthlyData[monthKey] = { income: 0, expense: 0 };
      }
      if (t.type === 'income') {
        monthlyData[monthKey].income += Number(t.amount);
      } else {
        monthlyData[monthKey].expense += Number(t.amount);
      }
    });

    const monthValues = Object.values(monthlyData);
    const avgIncome = monthValues.length > 0
      ? monthValues.reduce((a, b) => a + b.income, 0) / monthValues.length
      : 0;
    const avgExpense = monthValues.length > 0
      ? monthValues.reduce((a, b) => a + b.expense, 0) / monthValues.length
      : 0;

    // Project future cash flow
    const projections: any[] = [];
    let runningBalance = avgIncome - avgExpense; // Start with average net

    const currentDate = new Date();
    for (let i = 1; i <= months; i++) {
      const futureDate = new Date(currentDate);
      futureDate.setMonth(futureDate.getMonth() + i);
      
      const projectedIncome = avgIncome * (1 + (Math.random() * 0.05 - 0.025));
      const projectedExpense = avgExpense * (1 + (Math.random() * 0.05 - 0.025));
      const netCashFlow = projectedIncome - projectedExpense;
      runningBalance += netCashFlow;

      projections.push({
        month: `${futureDate.getFullYear()}-${String(futureDate.getMonth() + 1).padStart(2, '0')}`,
        projectedIncome: Math.round(projectedIncome),
        projectedExpense: Math.round(projectedExpense),
        netCashFlow: Math.round(netCashFlow),
        runningBalance: Math.round(runningBalance),
      });
    }

    res.status(200).json({
      success: true,
      data: {
        currentAverages: {
          income: Math.round(avgIncome),
          expense: Math.round(avgExpense),
          netMonthly: Math.round(avgIncome - avgExpense),
        },
        projections,
        savingsRate: avgIncome > 0 ? Math.round(((avgIncome - avgExpense) / avgIncome) * 100) : 0,
      },
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Get personalized nudges
// @route   GET /api/predictions/nudges
// @access  Private
export const getNudges = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const userId = req.user?.id;
    const nudges = await generateNudges(userId!);

    res.status(200).json({
      success: true,
      data: nudges,
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Get financial health score
// @route   GET /api/predictions/health-score
// @access  Private
export const getFinancialHealthScore = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const userId = req.user?.id;

    // Get last 3 months of data
    const threeMonthsAgo = new Date();
    threeMonthsAgo.setMonth(threeMonthsAgo.getMonth() - 3);

    const transactions = await Transaction.findAll({
      where: {
        userId,
        date: { [Op.gte]: threeMonthsAgo },
      },
    });

    const totalIncome = transactions
      .filter((t) => t.type === 'income')
      .reduce((sum, t) => sum + Number(t.amount), 0);
    
    const totalExpense = transactions
      .filter((t) => t.type === 'expense')
      .reduce((sum, t) => sum + Number(t.amount), 0);

    // Calculate scores (0-100)
    const savingsRate = totalIncome > 0 ? (totalIncome - totalExpense) / totalIncome : 0;
    const savingsScore = Math.min(100, Math.max(0, savingsRate * 200)); // 50% savings = 100 score

    // Diversity score - how spread out are expenses across categories
    const categorySpending: { [key: string]: number } = {};
    transactions
      .filter((t) => t.type === 'expense')
      .forEach((t) => {
        const cat = t.category || 'Other';
        categorySpending[cat] = (categorySpending[cat] || 0) + Number(t.amount);
      });
    
    const categories = Object.keys(categorySpending);
    const diversityScore = Math.min(100, categories.length * 10);

    // Consistency score - regular income
    const incomeTransactions = transactions.filter((t) => t.type === 'income');
    const consistencyScore = incomeTransactions.length >= 3 ? 80 : incomeTransactions.length * 25;

    // Overall score
    const overallScore = Math.round((savingsScore * 0.5) + (diversityScore * 0.2) + (consistencyScore * 0.3));

    res.status(200).json({
      success: true,
      data: {
        overallScore,
        breakdown: {
          savingsScore: Math.round(savingsScore),
          diversityScore: Math.round(diversityScore),
          consistencyScore: Math.round(consistencyScore),
        },
        metrics: {
          savingsRate: Math.round(savingsRate * 100),
          totalIncome,
          totalExpense,
          netSavings: totalIncome - totalExpense,
          categoriesUsed: categories.length,
        },
        recommendations: getHealthRecommendations(overallScore, savingsRate, categories.length),
      },
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

function getHealthRecommendations(score: number, savingsRate: number, categoryCount: number): string[] {
  const recommendations: string[] = [];

  if (savingsRate < 0.1) {
    recommendations.push('Try to save at least 10% of your income each month');
  } else if (savingsRate < 0.2) {
    recommendations.push('Great start! Aim to increase savings to 20% for better financial security');
  }

  if (categoryCount < 5) {
    recommendations.push('Track more expense categories for better insights');
  }

  if (score < 50) {
    recommendations.push('Focus on reducing non-essential expenses');
    recommendations.push('Set up a budget to control spending');
  } else if (score < 75) {
    recommendations.push('Consider setting up emergency fund goals');
  } else {
    recommendations.push('Excellent financial health! Consider investing your savings');
  }

  return recommendations;
}
