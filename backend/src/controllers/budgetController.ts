import { Response } from 'express';
import { Op } from 'sequelize';
import Budget from '../models/Budget';
import Transaction from '../models/Transaction';
import { AuthRequest } from '../middleware/auth';

// @desc    Get all budgets for user
// @route   GET /api/budgets
// @access  Private
export const getBudgets = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const { isActive } = req.query;
    const where: any = { userId: req.user?.id };

    if (isActive !== undefined) {
      where.isActive = isActive === 'true';
    }

    const budgets = await Budget.findAll({
      where,
      order: [['createdAt', 'DESC']],
    });

    // Calculate spent amount for each budget
    const budgetsWithSpent = await Promise.all(
      budgets.map(async (budget) => {
        const startDate = budget.startDate;
        const endDate = budget.endDate || new Date();

        const spent = await Transaction.sum('amount', {
          where: {
            userId: req.user?.id,
            type: 'expense',
            category: budget.category,
            date: {
              [Op.gte]: startDate,
              [Op.lte]: endDate,
            },
          },
        });

        return {
          ...budget.toJSON(),
          spent: spent || 0,
          remaining: Number(budget.amount) - (spent || 0),
          percentUsed: ((spent || 0) / Number(budget.amount)) * 100,
        };
      })
    );

    res.status(200).json({
      success: true,
      count: budgetsWithSpent.length,
      data: budgetsWithSpent,
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Get single budget
// @route   GET /api/budgets/:id
// @access  Private
export const getBudget = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const budget = await Budget.findOne({
      where: {
        id: req.params.id,
        userId: req.user?.id,
      },
    });

    if (!budget) {
      res.status(404).json({
        success: false,
        message: 'Budget not found',
      });
      return;
    }

    // Calculate spent
    const spent = await Transaction.sum('amount', {
      where: {
        userId: req.user?.id,
        type: 'expense',
        category: budget.category,
        date: {
          [Op.gte]: budget.startDate,
          [Op.lte]: budget.endDate || new Date(),
        },
      },
    });

    res.status(200).json({
      success: true,
      data: {
        ...budget.toJSON(),
        spent: spent || 0,
        remaining: Number(budget.amount) - (spent || 0),
        percentUsed: ((spent || 0) / Number(budget.amount)) * 100,
      },
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Create budget
// @route   POST /api/budgets
// @access  Private
export const createBudget = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const { category, amount, period, startDate, endDate, alertThreshold, notes } = req.body;

    // Check if budget for this category already exists
    const existingBudget = await Budget.findOne({
      where: {
        userId: req.user?.id,
        category,
        isActive: true,
      },
    });

    if (existingBudget) {
      res.status(400).json({
        success: false,
        message: 'Active budget for this category already exists',
      });
      return;
    }

    const budget = await Budget.create({
      userId: req.user?.id,
      category,
      amount,
      period: period || 'monthly',
      startDate: startDate || new Date(),
      endDate,
      alertThreshold: alertThreshold || 80,
      notes,
    });

    res.status(201).json({
      success: true,
      data: budget,
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Update budget
// @route   PUT /api/budgets/:id
// @access  Private
export const updateBudget = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const budget = await Budget.findOne({
      where: {
        id: req.params.id,
        userId: req.user?.id,
      },
    });

    if (!budget) {
      res.status(404).json({
        success: false,
        message: 'Budget not found',
      });
      return;
    }

    await budget.update(req.body);

    res.status(200).json({
      success: true,
      data: budget,
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Delete budget
// @route   DELETE /api/budgets/:id
// @access  Private
export const deleteBudget = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const budget = await Budget.findOne({
      where: {
        id: req.params.id,
        userId: req.user?.id,
      },
    });

    if (!budget) {
      res.status(404).json({
        success: false,
        message: 'Budget not found',
      });
      return;
    }

    await budget.destroy();

    res.status(200).json({
      success: true,
      message: 'Budget deleted successfully',
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Get budget summary
// @route   GET /api/budgets/summary
// @access  Private
export const getBudgetSummary = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const budgets = await Budget.findAll({
      where: {
        userId: req.user?.id,
        isActive: true,
      },
    });

    let totalBudget = 0;
    let totalSpent = 0;
    const categoryBreakdown: any[] = [];

    for (const budget of budgets) {
      const spent = await Transaction.sum('amount', {
        where: {
          userId: req.user?.id,
          type: 'expense',
          category: budget.category,
          date: {
            [Op.gte]: budget.startDate,
            [Op.lte]: budget.endDate || new Date(),
          },
        },
      }) || 0;

      totalBudget += Number(budget.amount);
      totalSpent += spent;

      categoryBreakdown.push({
        category: budget.category,
        budgeted: Number(budget.amount),
        spent,
        remaining: Number(budget.amount) - spent,
        percentUsed: (spent / Number(budget.amount)) * 100,
        status: spent > Number(budget.amount) ? 'exceeded' : 
                spent >= Number(budget.amount) * 0.8 ? 'warning' : 'good',
      });
    }

    res.status(200).json({
      success: true,
      data: {
        totalBudget,
        totalSpent,
        totalRemaining: totalBudget - totalSpent,
        overallPercentUsed: totalBudget > 0 ? (totalSpent / totalBudget) * 100 : 0,
        budgetCount: budgets.length,
        categoryBreakdown,
      },
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};
