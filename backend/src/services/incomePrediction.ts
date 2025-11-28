/**
 * Income Prediction Service
 * Uses statistical methods to predict future income based on historical patterns
 * 
 * In production, this would integrate with ML models (LSTM, Prophet)
 * For now, we use statistical analysis for predictions
 */

import { Op } from 'sequelize';
import Transaction from '../models/Transaction';
import sequelize from '../config/database';

interface IncomePrediction {
  predictedAmount: number;
  confidence: number;
  trend: 'increasing' | 'decreasing' | 'stable';
  seasonalPattern: string | null;
  volatility: number;
  recommendations: string[];
}

interface MonthlyIncomeData {
  year: number;
  month: number;
  total: number;
  count: number;
  avgAmount: number;
}

interface WeeklyPattern {
  dayOfWeek: number;
  avgAmount: number;
  frequency: number;
}

/**
 * Calculate standard deviation
 */
function calculateStdDev(values: number[], mean: number): number {
  if (values.length === 0) return 0;
  const squaredDiffs = values.map(value => Math.pow(value - mean, 2));
  const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
  return Math.sqrt(avgSquaredDiff);
}

/**
 * Calculate coefficient of variation (volatility measure)
 */
function calculateVolatility(values: number[]): number {
  if (values.length === 0) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  if (mean === 0) return 0;
  const stdDev = calculateStdDev(values, mean);
  return (stdDev / mean) * 100;
}

/**
 * Detect trend using linear regression
 */
function detectTrend(values: number[]): { slope: number; trend: 'increasing' | 'decreasing' | 'stable' } {
  if (values.length < 2) return { slope: 0, trend: 'stable' };

  const n = values.length;
  const xMean = (n - 1) / 2;
  const yMean = values.reduce((a, b) => a + b, 0) / n;

  let numerator = 0;
  let denominator = 0;

  for (let i = 0; i < n; i++) {
    numerator += (i - xMean) * (values[i] - yMean);
    denominator += Math.pow(i - xMean, 2);
  }

  const slope = denominator !== 0 ? numerator / denominator : 0;
  const percentChange = yMean !== 0 ? (slope / yMean) * 100 : 0;

  let trend: 'increasing' | 'decreasing' | 'stable';
  if (percentChange > 5) {
    trend = 'increasing';
  } else if (percentChange < -5) {
    trend = 'decreasing';
  } else {
    trend = 'stable';
  }

  return { slope, trend };
}

/**
 * Simple moving average prediction
 */
function movingAveragePrediction(values: number[], periods: number = 3): number {
  if (values.length === 0) return 0;
  const recentValues = values.slice(-periods);
  return recentValues.reduce((a, b) => a + b, 0) / recentValues.length;
}

/**
 * Weighted moving average (more weight to recent data)
 */
function weightedMovingAverage(values: number[]): number {
  if (values.length === 0) return 0;
  
  const weights = values.map((_, i) => i + 1);
  const weightSum = weights.reduce((a, b) => a + b, 0);
  
  let weightedSum = 0;
  for (let i = 0; i < values.length; i++) {
    weightedSum += values[i] * weights[i];
  }
  
  return weightedSum / weightSum;
}

/**
 * Detect seasonal patterns (monthly)
 */
function detectSeasonalPattern(monthlyData: MonthlyIncomeData[]): string | null {
  if (monthlyData.length < 6) return null;

  const monthlyAverages: { [key: number]: number[] } = {};
  
  monthlyData.forEach(data => {
    if (!monthlyAverages[data.month]) {
      monthlyAverages[data.month] = [];
    }
    monthlyAverages[data.month].push(data.total);
  });

  const avgByMonth: { month: number; avg: number }[] = [];
  for (const [month, values] of Object.entries(monthlyAverages)) {
    avgByMonth.push({
      month: parseInt(month),
      avg: values.reduce((a, b) => a + b, 0) / values.length,
    });
  }

  if (avgByMonth.length < 4) return null;

  const overallAvg = avgByMonth.reduce((sum, m) => sum + m.avg, 0) / avgByMonth.length;
  const highMonths = avgByMonth.filter(m => m.avg > overallAvg * 1.2);
  const lowMonths = avgByMonth.filter(m => m.avg < overallAvg * 0.8);

  if (highMonths.length === 0 && lowMonths.length === 0) {
    return null;
  }

  const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  
  let pattern = '';
  if (highMonths.length > 0) {
    pattern += `High income: ${highMonths.map(m => monthNames[m.month - 1]).join(', ')}. `;
  }
  if (lowMonths.length > 0) {
    pattern += `Low income: ${lowMonths.map(m => monthNames[m.month - 1]).join(', ')}.`;
  }

  return pattern.trim();
}

/**
 * Generate recommendations based on income patterns
 */
function generateRecommendations(
  volatility: number,
  trend: 'increasing' | 'decreasing' | 'stable',
  predictedAmount: number,
  avgAmount: number,
  seasonalPattern: string | null
): string[] {
  const recommendations: string[] = [];

  // Volatility-based recommendations
  if (volatility > 50) {
    recommendations.push('Your income is highly variable. Build an emergency fund covering 6 months of expenses.');
    recommendations.push('Consider diversifying income sources to reduce variability.');
  } else if (volatility > 30) {
    recommendations.push('Your income shows moderate variability. Maintain 3-4 months of expenses as emergency fund.');
  }

  // Trend-based recommendations
  if (trend === 'decreasing') {
    recommendations.push('Your income trend is declining. Review your client pipeline or work opportunities.');
    recommendations.push('Consider reducing non-essential expenses during this period.');
  } else if (trend === 'increasing') {
    recommendations.push('Great job! Your income is growing. Consider increasing your savings rate.');
    recommendations.push('This is a good time to accelerate debt repayment or goal savings.');
  }

  // Prediction-based recommendations
  if (predictedAmount < avgAmount * 0.8) {
    recommendations.push(`Next month's income may be lower than average. Prepare by reducing discretionary spending.`);
  } else if (predictedAmount > avgAmount * 1.2) {
    recommendations.push(`Higher income expected next month. Allocate extra towards goals or emergency fund.`);
  }

  // Seasonal recommendations
  if (seasonalPattern) {
    recommendations.push(`Seasonal pattern detected: ${seasonalPattern} Plan your budget accordingly.`);
  }

  return recommendations;
}

/**
 * Get income prediction for a user
 */
export async function predictIncome(userId: number, _monthsAhead: number = 1): Promise<IncomePrediction> {
  // Get last 12 months of income data
  const twelveMonthsAgo = new Date();
  twelveMonthsAgo.setMonth(twelveMonthsAgo.getMonth() - 12);

  const monthlyIncome = await Transaction.findAll({
    where: {
      userId,
      type: 'income',
      date: { [Op.gte]: twelveMonthsAgo },
    },
    attributes: [
      [sequelize.fn('EXTRACT', sequelize.literal('YEAR FROM date')), 'year'],
      [sequelize.fn('EXTRACT', sequelize.literal('MONTH FROM date')), 'month'],
      [sequelize.fn('SUM', sequelize.col('amount')), 'total'],
      [sequelize.fn('COUNT', sequelize.col('id')), 'count'],
      [sequelize.fn('AVG', sequelize.col('amount')), 'avgAmount'],
    ],
    group: [
      sequelize.fn('EXTRACT', sequelize.literal('YEAR FROM date')),
      sequelize.fn('EXTRACT', sequelize.literal('MONTH FROM date')),
    ],
    order: [
      [sequelize.fn('EXTRACT', sequelize.literal('YEAR FROM date')), 'ASC'],
      [sequelize.fn('EXTRACT', sequelize.literal('MONTH FROM date')), 'ASC'],
    ],
    raw: true,
  }) as unknown as MonthlyIncomeData[];

  // Handle case with no data
  if (monthlyIncome.length === 0) {
    return {
      predictedAmount: 0,
      confidence: 0,
      trend: 'stable',
      seasonalPattern: null,
      volatility: 0,
      recommendations: ['Start tracking your income to get predictions.'],
    };
  }

  const incomeValues = monthlyIncome.map(m => Number(m.total));
  const avgIncome = incomeValues.reduce((a, b) => a + b, 0) / incomeValues.length;

  // Calculate volatility
  const volatility = calculateVolatility(incomeValues);

  // Detect trend
  const { trend } = detectTrend(incomeValues);

  // Calculate predictions using ensemble of methods
  const smaPredict = movingAveragePrediction(incomeValues, 3);
  const wmaPredict = weightedMovingAverage(incomeValues);
  
  // Combine predictions (weighted average of methods)
  let predictedAmount = (smaPredict * 0.4 + wmaPredict * 0.6);

  // Adjust for trend
  if (trend === 'increasing') {
    predictedAmount *= 1.05;
  } else if (trend === 'decreasing') {
    predictedAmount *= 0.95;
  }

  // Calculate confidence based on data quality and volatility
  let confidence = 100;
  if (monthlyIncome.length < 3) confidence -= 30;
  if (monthlyIncome.length < 6) confidence -= 20;
  if (volatility > 50) confidence -= 25;
  else if (volatility > 30) confidence -= 15;
  confidence = Math.max(confidence, 20);

  // Detect seasonal patterns
  const seasonalPattern = detectSeasonalPattern(monthlyIncome);

  // Generate recommendations
  const recommendations = generateRecommendations(
    volatility,
    trend,
    predictedAmount,
    avgIncome,
    seasonalPattern
  );

  return {
    predictedAmount: Number(predictedAmount.toFixed(2)),
    confidence: Number(confidence.toFixed(0)),
    trend,
    seasonalPattern,
    volatility: Number(volatility.toFixed(1)),
    recommendations,
  };
}

/**
 * Get weekly income patterns
 */
export async function getWeeklyPatterns(userId: number): Promise<WeeklyPattern[]> {
  const threeMonthsAgo = new Date();
  threeMonthsAgo.setMonth(threeMonthsAgo.getMonth() - 3);

  const weeklyData = await Transaction.findAll({
    where: {
      userId,
      type: 'income',
      date: { [Op.gte]: threeMonthsAgo },
    },
    attributes: [
      [sequelize.fn('EXTRACT', sequelize.literal('DOW FROM date')), 'dayOfWeek'],
      [sequelize.fn('AVG', sequelize.col('amount')), 'avgAmount'],
      [sequelize.fn('COUNT', sequelize.col('id')), 'frequency'],
    ],
    group: [sequelize.fn('EXTRACT', sequelize.literal('DOW FROM date'))],
    order: [[sequelize.fn('EXTRACT', sequelize.literal('DOW FROM date')), 'ASC']],
    raw: true,
  }) as unknown as WeeklyPattern[];

  return weeklyData.map(d => ({
    dayOfWeek: Number(d.dayOfWeek),
    avgAmount: Number(Number(d.avgAmount).toFixed(2)),
    frequency: Number(d.frequency),
  }));
}

/**
 * Get income variability analysis
 */
export async function analyzeIncomeVariability(userId: number): Promise<{
  monthlyStats: { min: number; max: number; avg: number; median: number };
  variabilityScore: 'low' | 'moderate' | 'high' | 'very_high';
  incomeStability: number;
  suggestions: string[];
}> {
  const sixMonthsAgo = new Date();
  sixMonthsAgo.setMonth(sixMonthsAgo.getMonth() - 6);

  const monthlyIncome = await Transaction.findAll({
    where: {
      userId,
      type: 'income',
      date: { [Op.gte]: sixMonthsAgo },
    },
    attributes: [
      [sequelize.fn('EXTRACT', sequelize.literal('YEAR FROM date')), 'year'],
      [sequelize.fn('EXTRACT', sequelize.literal('MONTH FROM date')), 'month'],
      [sequelize.fn('SUM', sequelize.col('amount')), 'total'],
    ],
    group: [
      sequelize.fn('EXTRACT', sequelize.literal('YEAR FROM date')),
      sequelize.fn('EXTRACT', sequelize.literal('MONTH FROM date')),
    ],
    raw: true,
  }) as unknown as { year: number; month: number; total: number }[];

  if (monthlyIncome.length === 0) {
    return {
      monthlyStats: { min: 0, max: 0, avg: 0, median: 0 },
      variabilityScore: 'low',
      incomeStability: 100,
      suggestions: ['Start tracking income to get variability analysis.'],
    };
  }

  const values = monthlyIncome.map(m => Number(m.total)).sort((a, b) => a - b);
  const min = values[0];
  const max = values[values.length - 1];
  const avg = values.reduce((a, b) => a + b, 0) / values.length;
  const median = values.length % 2 === 0
    ? (values[values.length / 2 - 1] + values[values.length / 2]) / 2
    : values[Math.floor(values.length / 2)];

  const volatility = calculateVolatility(values);
  
  let variabilityScore: 'low' | 'moderate' | 'high' | 'very_high';
  let incomeStability: number;
  const suggestions: string[] = [];

  if (volatility <= 15) {
    variabilityScore = 'low';
    incomeStability = 90;
    suggestions.push('Your income is very stable. This is ideal for consistent budgeting.');
  } else if (volatility <= 30) {
    variabilityScore = 'moderate';
    incomeStability = 70;
    suggestions.push('Your income has moderate variability. Budget based on your minimum expected income.');
  } else if (volatility <= 50) {
    variabilityScore = 'high';
    incomeStability = 50;
    suggestions.push('Your income is highly variable. Maintain a larger emergency fund (4-6 months expenses).');
    suggestions.push('Consider income smoothing: save more in high months, spend from savings in low months.');
  } else {
    variabilityScore = 'very_high';
    incomeStability = 30;
    suggestions.push('Your income is extremely variable. Build emergency fund covering 6+ months of expenses.');
    suggestions.push('Consider diversifying income sources to reduce variability.');
    suggestions.push('Budget based on your lowest income month to avoid overspending.');
  }

  return {
    monthlyStats: {
      min: Number(min.toFixed(2)),
      max: Number(max.toFixed(2)),
      avg: Number(avg.toFixed(2)),
      median: Number(median.toFixed(2)),
    },
    variabilityScore,
    incomeStability,
    suggestions,
  };
}
