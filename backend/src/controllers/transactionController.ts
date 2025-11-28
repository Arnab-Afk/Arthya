import { Response } from 'express';
import { Op } from 'sequelize';
import Transaction from '../models/Transaction';
import { AuthRequest } from '../middleware/auth';
import sequelize from '../config/database';
import { categorizeTransaction } from '../services/categorization';

// @desc    Get all transactions for user
// @route   GET /api/transactions
// @access  Private
export const getTransactions = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const { type, startDate, endDate, category, limit = 50, page = 1 } = req.query;

    const where: any = { userId: req.user?.id };

    if (type) where.type = type;
    if (category) where.category = category;
    if (startDate || endDate) {
      where.date = {};
      if (startDate) where.date[Op.gte] = new Date(startDate as string);
      if (endDate) where.date[Op.lte] = new Date(endDate as string);
    }

    const offset = (Number(page) - 1) * Number(limit);

    const { count, rows: transactions } = await Transaction.findAndCountAll({
      where,
      order: [['date', 'DESC']],
      limit: Number(limit),
      offset,
    });

    res.status(200).json({
      success: true,
      count: transactions.length,
      total: count,
      page: Number(page),
      pages: Math.ceil(count / Number(limit)),
      data: transactions,
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Get single transaction
// @route   GET /api/transactions/:id
// @access  Private
export const getTransaction = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const transaction = await Transaction.findOne({
      where: {
        id: req.params.id,
        userId: req.user?.id,
      },
    });

    if (!transaction) {
      res.status(404).json({
        success: false,
        message: 'Transaction not found',
      });
      return;
    }

    res.status(200).json({
      success: true,
      data: transaction,
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Create transaction
// @route   POST /api/transactions
// @access  Private
export const createTransaction = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const transactionData = {
      ...req.body,
      userId: req.user?.id,
    };

    const transaction = await Transaction.create(transactionData);

    res.status(201).json({
      success: true,
      data: transaction,
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Update transaction
// @route   PUT /api/transactions/:id
// @access  Private
export const updateTransaction = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const transaction = await Transaction.findOne({
      where: {
        id: req.params.id,
        userId: req.user?.id,
      },
    });

    if (!transaction) {
      res.status(404).json({
        success: false,
        message: 'Transaction not found',
      });
      return;
    }

    await transaction.update(req.body);

    res.status(200).json({
      success: true,
      data: transaction,
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Delete transaction
// @route   DELETE /api/transactions/:id
// @access  Private
export const deleteTransaction = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const transaction = await Transaction.findOne({
      where: {
        id: req.params.id,
        userId: req.user?.id,
      },
    });

    if (!transaction) {
      res.status(404).json({
        success: false,
        message: 'Transaction not found',
      });
      return;
    }

    await transaction.destroy();

    res.status(200).json({
      success: true,
      message: 'Transaction deleted successfully',
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Get transaction summary
// @route   GET /api/transactions/summary
// @access  Private
export const getTransactionSummary = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const { startDate, endDate } = req.query;
    const where: any = { userId: req.user?.id };

    if (startDate || endDate) {
      where.date = {};
      if (startDate) where.date[Op.gte] = new Date(startDate as string);
      if (endDate) where.date[Op.lte] = new Date(endDate as string);
    }

    const summary = await Transaction.findAll({
      where,
      attributes: [
        'type',
        [sequelize.fn('SUM', sequelize.col('amount')), 'total'],
        [sequelize.fn('COUNT', sequelize.col('id')), 'count'],
      ],
      group: ['type'],
      raw: true,
    });

    const formattedSummary = {
      income: 0,
      expense: 0,
      transfer: 0,
      incomeCount: 0,
      expenseCount: 0,
      transferCount: 0,
    };

    summary.forEach((item: any) => {
      const total = parseFloat(item.total);
      const count = parseInt(item.count);

      if (item.type === 'income') {
        formattedSummary.income = total;
        formattedSummary.incomeCount = count;
      } else if (item.type === 'expense') {
        formattedSummary.expense = total;
        formattedSummary.expenseCount = count;
      } else if (item.type === 'transfer') {
        formattedSummary.transfer = total;
        formattedSummary.transferCount = count;
      }
    });

    formattedSummary.income = Number(formattedSummary.income.toFixed(2));
    formattedSummary.expense = Number(formattedSummary.expense.toFixed(2));
    formattedSummary.transfer = Number(formattedSummary.transfer.toFixed(2));

    res.status(200).json({
      success: true,
      data: {
        ...formattedSummary,
        savings: Number((formattedSummary.income - formattedSummary.expense).toFixed(2)),
      },
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Bulk import transactions (from SMS)
// @route   POST /api/transactions/bulk-import
// @access  Private
export const bulkImportTransactions = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const { transactions } = req.body;

    if (!Array.isArray(transactions) || transactions.length === 0) {
      res.status(400).json({
        success: false,
        message: 'Transactions array is required',
      });
      return;
    }

    const userId = req.user?.id;
    const results = {
      imported: 0,
      skipped: 0,
      errors: [] as string[],
    };

    for (const txn of transactions) {
      try {
        // Check for duplicates based on amount, date, and description
        const existingTxn = await Transaction.findOne({
          where: {
            userId,
            amount: txn.amount,
            description: txn.description,
            date: {
              [Op.between]: [
                new Date(new Date(txn.date).getTime() - 60000), // 1 minute before
                new Date(new Date(txn.date).getTime() + 60000), // 1 minute after
              ],
            },
          },
        });

        if (existingTxn) {
          results.skipped++;
          continue;
        }

        // Auto-categorize if no category provided
        let category = txn.category;
        if (!category && txn.description) {
          const categorized = categorizeTransaction(txn.description, txn.amount);
          category = categorized.category;
        }

        await Transaction.create({
          userId,
          type: txn.type || 'expense',
          amount: txn.amount,
          description: txn.description,
          category: category || 'Other',
          date: new Date(txn.date),
          source: txn.source || 'sms_import',
          notes: txn.notes,
        });

        results.imported++;
      } catch (err: any) {
        results.errors.push(`Failed to import: ${txn.description} - ${err.message}`);
      }
    }

    res.status(200).json({
      success: true,
      data: {
        ...results,
        message: `Successfully imported ${results.imported} transactions, skipped ${results.skipped} duplicates`,
      },
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Get spending by category
// @route   GET /api/transactions/by-category
// @access  Private
export const getSpendingByCategory = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const { startDate, endDate } = req.query;
    const where: any = { 
      userId: req.user?.id,
      type: 'expense',
    };

    if (startDate || endDate) {
      where.date = {};
      if (startDate) where.date[Op.gte] = new Date(startDate as string);
      if (endDate) where.date[Op.lte] = new Date(endDate as string);
    }

    const categorySpending = await Transaction.findAll({
      where,
      attributes: [
        'category',
        [sequelize.fn('SUM', sequelize.col('amount')), 'total'],
        [sequelize.fn('COUNT', sequelize.col('id')), 'count'],
      ],
      group: ['category'],
      order: [[sequelize.literal('total'), 'DESC']],
      raw: true,
    });

    const total = categorySpending.reduce((sum: number, item: any) => sum + parseFloat(item.total), 0);

    const formattedData = categorySpending.map((item: any) => ({
      category: item.category || 'Uncategorized',
      total: parseFloat(item.total),
      count: parseInt(item.count),
      percentage: total > 0 ? ((parseFloat(item.total) / total) * 100).toFixed(1) : 0,
    }));

    res.status(200).json({
      success: true,
      data: {
        categories: formattedData,
        totalSpending: total,
      },
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Get monthly trends
// @route   GET /api/transactions/monthly-trends
// @access  Private
export const getMonthlyTrends = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const months = parseInt(req.query.months as string) || 6;
    const startDate = new Date();
    startDate.setMonth(startDate.getMonth() - months);

    const transactions = await Transaction.findAll({
      where: {
        userId: req.user?.id,
        date: { [Op.gte]: startDate },
      },
      order: [['date', 'ASC']],
    });

    // Group by month
    const monthlyData: { [key: string]: { income: number; expense: number } } = {};
    
    transactions.forEach((txn) => {
      const monthKey = `${txn.date.getFullYear()}-${String(txn.date.getMonth() + 1).padStart(2, '0')}`;
      if (!monthlyData[monthKey]) {
        monthlyData[monthKey] = { income: 0, expense: 0 };
      }
      if (txn.type === 'income') {
        monthlyData[monthKey].income += Number(txn.amount);
      } else if (txn.type === 'expense') {
        monthlyData[monthKey].expense += Number(txn.amount);
      }
    });

    const trends = Object.entries(monthlyData).map(([month, data]) => ({
      month,
      income: Math.round(data.income),
      expense: Math.round(data.expense),
      savings: Math.round(data.income - data.expense),
      savingsRate: data.income > 0 ? Math.round(((data.income - data.expense) / data.income) * 100) : 0,
    }));

    res.status(200).json({
      success: true,
      data: trends,
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};
