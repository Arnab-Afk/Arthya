import { Response } from 'express';
import Achievement from '../models/Achievement';
import Transaction from '../models/Transaction';
import Goal from '../models/Goal';
import { AuthRequest } from '../middleware/auth';
// Sequelize operators imported when needed

// Achievement definitions
const ACHIEVEMENTS = [
  {
    type: 'first_transaction',
    title: 'First Step',
    description: 'Record your first transaction',
    icon: '🎯',
    category: 'onboarding',
  },
  {
    type: 'week_streak',
    title: 'Weekly Warrior',
    description: 'Log transactions for 7 consecutive days',
    icon: '🔥',
    category: 'consistency',
  },
  {
    type: 'month_streak',
    title: 'Monthly Master',
    description: 'Log transactions for 30 consecutive days',
    icon: '⚡',
    category: 'consistency',
  },
  {
    type: 'first_goal',
    title: 'Goal Setter',
    description: 'Create your first savings goal',
    icon: '🎯',
    category: 'goals',
  },
  {
    type: 'goal_achieved',
    title: 'Goal Crusher',
    description: 'Complete your first savings goal',
    icon: '🏆',
    category: 'goals',
  },
  {
    type: 'budget_created',
    title: 'Budget Boss',
    description: 'Create your first budget',
    icon: '💰',
    category: 'budgeting',
  },
  {
    type: 'under_budget',
    title: 'Budget Keeper',
    description: 'Stay under budget for a full month',
    icon: '✅',
    category: 'budgeting',
  },
  {
    type: 'saver_bronze',
    title: 'Bronze Saver',
    description: 'Save ₹1,000',
    icon: '🥉',
    category: 'savings',
  },
  {
    type: 'saver_silver',
    title: 'Silver Saver',
    description: 'Save ₹10,000',
    icon: '🥈',
    category: 'savings',
  },
  {
    type: 'saver_gold',
    title: 'Gold Saver',
    description: 'Save ₹50,000',
    icon: '🥇',
    category: 'savings',
  },
  {
    type: 'transaction_100',
    title: 'Century Club',
    description: 'Log 100 transactions',
    icon: '💯',
    category: 'activity',
  },
  {
    type: 'transaction_500',
    title: 'Power Tracker',
    description: 'Log 500 transactions',
    icon: '🚀',
    category: 'activity',
  },
  {
    type: 'category_explorer',
    title: 'Category Explorer',
    description: 'Use 10 different expense categories',
    icon: '🗺️',
    category: 'exploration',
  },
];

// @desc    Get all achievements for user
// @route   GET /api/achievements
// @access  Private
export const getAchievements = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    // Get unlocked achievements
    const unlockedAchievements = await Achievement.findAll({
      where: { userId: req.user?.id },
      order: [['unlockedAt', 'DESC']],
    });

    // Map all achievements with unlocked status
    const allAchievements = ACHIEVEMENTS.map((achievement) => {
      const unlocked = unlockedAchievements.find(
        (a) => a.type === achievement.type
      );
      return {
        ...achievement,
        unlocked: !!unlocked,
        unlockedAt: unlocked?.unlockedAt || null,
        progress: 0, // Will be calculated per achievement type
      };
    });

    res.status(200).json({
      success: true,
      data: {
        achievements: allAchievements,
        totalUnlocked: unlockedAchievements.length,
        totalAchievements: ACHIEVEMENTS.length,
        completionPercentage:
          (unlockedAchievements.length / ACHIEVEMENTS.length) * 100,
      },
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// @desc    Check and unlock achievements
// @route   POST /api/achievements/check
// @access  Private
export const checkAchievements = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const userId = req.user?.id;
    const newlyUnlocked: any[] = [];

    // Get existing achievements
    const existingAchievements = await Achievement.findAll({
      where: { userId },
    });
    const unlockedTypes = existingAchievements.map((a) => a.type);

    // Check each achievement
    // 1. First transaction
    if (!unlockedTypes.includes('first_transaction')) {
      const transactionCount = await Transaction.count({ where: { userId } });
      if (transactionCount > 0) {
        const achievement = await unlockAchievement(userId!, 'first_transaction');
        if (achievement) newlyUnlocked.push(achievement);
      }
    }

    // 2. Transaction milestones
    const transactionCount = await Transaction.count({ where: { userId } });
    
    if (!unlockedTypes.includes('transaction_100') && transactionCount >= 100) {
      const achievement = await unlockAchievement(userId!, 'transaction_100');
      if (achievement) newlyUnlocked.push(achievement);
    }
    
    if (!unlockedTypes.includes('transaction_500') && transactionCount >= 500) {
      const achievement = await unlockAchievement(userId!, 'transaction_500');
      if (achievement) newlyUnlocked.push(achievement);
    }

    // 3. First goal
    if (!unlockedTypes.includes('first_goal')) {
      const goalCount = await Goal.count({ where: { userId } });
      if (goalCount > 0) {
        const achievement = await unlockAchievement(userId!, 'first_goal');
        if (achievement) newlyUnlocked.push(achievement);
      }
    }

    // 4. Goal achieved
    if (!unlockedTypes.includes('goal_achieved')) {
      const completedGoals = await Goal.count({
        where: { userId, status: 'completed' },
      });
      if (completedGoals > 0) {
        const achievement = await unlockAchievement(userId!, 'goal_achieved');
        if (achievement) newlyUnlocked.push(achievement);
      }
    }

    // 5. Savings milestones
    const totalSavings = await Transaction.sum('amount', {
      where: { userId, type: 'income' },
    }) || 0;
    
    const totalExpenses = await Transaction.sum('amount', {
      where: { userId, type: 'expense' },
    }) || 0;
    
    const netSavings = totalSavings - totalExpenses;

    if (!unlockedTypes.includes('saver_bronze') && netSavings >= 1000) {
      const achievement = await unlockAchievement(userId!, 'saver_bronze');
      if (achievement) newlyUnlocked.push(achievement);
    }
    
    if (!unlockedTypes.includes('saver_silver') && netSavings >= 10000) {
      const achievement = await unlockAchievement(userId!, 'saver_silver');
      if (achievement) newlyUnlocked.push(achievement);
    }
    
    if (!unlockedTypes.includes('saver_gold') && netSavings >= 50000) {
      const achievement = await unlockAchievement(userId!, 'saver_gold');
      if (achievement) newlyUnlocked.push(achievement);
    }

    // 6. Category explorer
    if (!unlockedTypes.includes('category_explorer')) {
      const categories = await Transaction.findAll({
        where: { userId, type: 'expense' },
        attributes: ['category'],
        group: ['category'],
      });
      if (categories.length >= 10) {
        const achievement = await unlockAchievement(userId!, 'category_explorer');
        if (achievement) newlyUnlocked.push(achievement);
      }
    }

    res.status(200).json({
      success: true,
      data: {
        newlyUnlocked,
        message:
          newlyUnlocked.length > 0
            ? `Congratulations! You unlocked ${newlyUnlocked.length} new achievement(s)!`
            : 'No new achievements unlocked',
      },
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};

// Helper function to unlock an achievement
async function unlockAchievement(
  userId: string,
  type: string
): Promise<any | null> {
  const achievementDef = ACHIEVEMENTS.find((a) => a.type === type);
  if (!achievementDef) return null;

  // Check if already unlocked
  const existing = await Achievement.findOne({
    where: { userId, type },
  });
  if (existing) return null;

  const achievement = await Achievement.create({
    userId,
    type,
    title: achievementDef.title,
    description: achievementDef.description,
    icon: achievementDef.icon,
    category: achievementDef.category,
    unlockedAt: new Date(),
  });

  return achievement;
}

// @desc    Get achievement progress
// @route   GET /api/achievements/progress
// @access  Private
export const getAchievementProgress = async (
  req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    const userId = req.user?.id;

    // Calculate progress for various achievements
    const transactionCount = await Transaction.count({ where: { userId } });
    
    const totalSavings = await Transaction.sum('amount', {
      where: { userId, type: 'income' },
    }) || 0;
    
    const totalExpenses = await Transaction.sum('amount', {
      where: { userId, type: 'expense' },
    }) || 0;
    
    const netSavings = totalSavings - totalExpenses;

    const categories = await Transaction.findAll({
      where: { userId, type: 'expense' },
      attributes: ['category'],
      group: ['category'],
    });

    const goals = await Goal.findAll({ where: { userId } });
    const completedGoals = goals.filter((g) => g.status === 'completed').length;

    res.status(200).json({
      success: true,
      data: {
        transactions: {
          current: transactionCount,
          next_milestone: transactionCount < 100 ? 100 : transactionCount < 500 ? 500 : 1000,
        },
        savings: {
          current: netSavings,
          next_milestone: netSavings < 1000 ? 1000 : netSavings < 10000 ? 10000 : 50000,
        },
        categories: {
          current: categories.length,
          target: 10,
        },
        goals: {
          created: goals.length,
          completed: completedGoals,
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

// @desc    Get leaderboard (optional gamification)
// @route   GET /api/achievements/leaderboard
// @access  Private
export const getLeaderboard = async (
  _req: AuthRequest,
  res: Response
): Promise<void> => {
  try {
    // This is a simplified leaderboard based on achievement count
    // In a real app, you'd want to add privacy controls
    const leaderboard = await Achievement.findAll({
      attributes: [
        'userId',
        [Achievement.sequelize!.fn('COUNT', Achievement.sequelize!.col('id')), 'achievementCount'],
      ],
      group: ['userId'],
      order: [[Achievement.sequelize!.literal('achievementCount'), 'DESC']],
      limit: 10,
    });

    res.status(200).json({
      success: true,
      data: leaderboard,
    });
  } catch (error: any) {
    res.status(400).json({
      success: false,
      message: error.message,
    });
  }
};
