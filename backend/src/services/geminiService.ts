import { GoogleGenerativeAI } from '@google/generative-ai';
import type { FinancialProfile } from './coachingAI';

// Initialize Gemini with API key from environment
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || '');

// Get the Gemini model - using gemini-2.0-flash which is the latest available
const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash' });

/**
 * Generate personalized financial advice using Gemini AI
 */
export async function generateAIAdvice(profile: FinancialProfile): Promise<{
  insights: Array<{
    id: string;
    type: 'tip' | 'warning' | 'celebration' | 'action' | 'education';
    category: string;
    priority: 'high' | 'medium' | 'low';
    title: string;
    message: string;
    actionItems: string[];
  }>;
  summary: string;
}> {
  const prompt = `You are an AI financial coach for gig workers in India. Analyze the following financial profile and provide personalized advice.

FINANCIAL PROFILE:
- Monthly Income: ₹${profile.income.total.toFixed(0)} (${profile.income.trend} trend)
- Income Volatility: ${profile.income.volatility.toFixed(1)}%
- Monthly Expenses: ₹${profile.expenses.total.toFixed(0)} (${profile.expenses.trend} trend)
- Top Expense Categories: ${profile.expenses.byCategory.slice(0, 3).map(c => `${c.category}: ₹${c.amount.toFixed(0)} (${c.percentage.toFixed(0)}%)`).join(', ')}
- Savings Rate: ${profile.savings.rate.toFixed(1)}%
- Net Savings: ₹${profile.savings.netAmount.toFixed(0)}
- Active Goals: ${profile.goals.active} (${profile.goals.atRisk} at risk)
- Budgets: ${profile.budgets.total} total (${profile.budgets.exceeded} exceeded)
- Spending Pattern: ${profile.behaviorPatterns.spendingDays.length > 0 ? `Heavy spending on ${profile.behaviorPatterns.spendingDays.join(', ')}` : 'Balanced throughout week'}

Based on this profile, provide:
1. 3-5 specific, actionable insights
2. A brief summary paragraph

Respond in this exact JSON format:
{
  "insights": [
    {
      "id": "unique_id",
      "type": "tip|warning|celebration|action|education",
      "category": "savings|spending|income|goals|budgets|habits",
      "priority": "high|medium|low",
      "title": "Short title",
      "message": "Detailed message in 1-2 sentences",
      "actionItems": ["Action 1", "Action 2"]
    }
  ],
  "summary": "Overall summary paragraph"
}

Focus on practical advice relevant to Indian gig workers. Use ₹ for currency. Be encouraging but honest.`;

  try {
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const text = response.text();
    
    // Extract JSON from the response
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[0]);
      return parsed;
    }
    
    // Fallback if parsing fails
    return getDefaultAdvice(profile);
  } catch (error) {
    console.error('Gemini API error:', error);
    return getDefaultAdvice(profile);
  }
}

/**
 * Chat with AI financial coach using Gemini
 */
export async function chatWithAI(
  message: string, 
  profile: FinancialProfile,
  conversationHistory: Array<{ role: 'user' | 'model'; content: string }> = []
): Promise<{
  response: string;
  suggestions: string[];
}> {
  const systemContext = `You are Arthya AI, a friendly and knowledgeable financial coach for gig workers in India. 

Current user's financial snapshot:
- Savings Rate: ${profile.savings.rate.toFixed(1)}%
- Top Expense: ${profile.expenses.byCategory[0]?.category || 'N/A'} (${profile.expenses.byCategory[0]?.percentage.toFixed(0) || 0}%)
- Active Goals: ${profile.goals.active}
- Budgets On Track: ${profile.budgets.onTrack}/${profile.budgets.total}
- Income Trend: ${profile.income.trend}

Guidelines:
- Be conversational, empathetic, and encouraging
- Give specific, actionable advice
- Use ₹ for currency amounts
- Keep responses concise (2-4 sentences max)
- Suggest follow-up topics when relevant
- If asked about investments, mention that you can provide general guidance but recommend consulting a financial advisor for specific investment decisions`;

  const prompt = `${systemContext}

${conversationHistory.length > 0 ? 'Previous conversation:\n' + conversationHistory.map(h => `${h.role === 'user' ? 'User' : 'Arthya'}: ${h.content}`).join('\n') + '\n\n' : ''}

User's message: "${message}"

Respond in JSON format:
{
  "response": "Your helpful response here",
  "suggestions": ["Follow-up topic 1", "Follow-up topic 2"]
}`;

  try {
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const text = response.text();
    
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[0]);
      return {
        response: parsed.response,
        suggestions: parsed.suggestions || []
      };
    }
    
    return {
      response: text.replace(/```json|```/g, '').trim(),
      suggestions: ['Tell me about saving tips', 'How are my goals?']
    };
  } catch (error) {
    console.error('Gemini chat error:', error);
    return getFallbackChatResponse(message, profile);
  }
}

/**
 * Generate weekly financial summary using Gemini
 */
export async function generateAIWeeklySummary(
  weeklyData: {
    income: number;
    expenses: number;
    savings: number;
    topCategories: Array<{ category: string; amount: number }>;
    comparedToLastWeek: { incomeChange: number; expenseChange: number };
  },
  profile: FinancialProfile
): Promise<{
  headline: string;
  highlights: string[];
  tipOfTheWeek: string;
  encouragement: string;
}> {
  const prompt = `As a financial coach, create a brief weekly summary for a gig worker in India.

THIS WEEK'S DATA:
- Income: ₹${weeklyData.income}
- Expenses: ₹${weeklyData.expenses}
- Savings: ₹${weeklyData.savings}
- Top spending: ${weeklyData.topCategories.map(c => `${c.category}: ₹${c.amount}`).join(', ')}
- Income vs last week: ${weeklyData.comparedToLastWeek.incomeChange >= 0 ? '+' : ''}${weeklyData.comparedToLastWeek.incomeChange.toFixed(1)}%
- Expenses vs last week: ${weeklyData.comparedToLastWeek.expenseChange >= 0 ? '+' : ''}${weeklyData.comparedToLastWeek.expenseChange.toFixed(1)}%

OVERALL CONTEXT:
- Monthly savings rate: ${profile.savings.rate.toFixed(1)}%
- Active goals: ${profile.goals.active}

Respond in JSON format:
{
  "headline": "One catchy line summarizing the week (use emojis)",
  "highlights": ["Key point 1", "Key point 2", "Key point 3"],
  "tipOfTheWeek": "One practical tip based on this week's data",
  "encouragement": "A short encouraging message"
}`;

  try {
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const text = response.text();
    
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      return JSON.parse(jsonMatch[0]);
    }
    
    return getDefaultWeeklySummary(weeklyData);
  } catch (error) {
    console.error('Gemini weekly summary error:', error);
    return getDefaultWeeklySummary(weeklyData);
  }
}

/**
 * Generate spending pattern analysis using Gemini
 */
export async function analyzeSpendingPatterns(
  transactions: Array<{ category: string; amount: number; date: Date; description: string }>,
  profile: FinancialProfile
): Promise<{
  patterns: Array<{ pattern: string; frequency: string; impact: string }>;
  recommendations: string[];
  savingsOpportunities: Array<{ area: string; potentialSavings: number; suggestion: string }>;
}> {
  // Group transactions by category and time
  const categoryTotals: { [key: string]: number } = {};
  const dayOfWeekSpending: { [key: number]: number } = {};
  
  transactions.forEach(t => {
    categoryTotals[t.category] = (categoryTotals[t.category] || 0) + t.amount;
    const day = new Date(t.date).getDay();
    dayOfWeekSpending[day] = (dayOfWeekSpending[day] || 0) + t.amount;
  });

  const prompt = `Analyze spending patterns for a gig worker in India.

SPENDING DATA:
- Total transactions: ${transactions.length}
- Categories: ${Object.entries(categoryTotals).map(([cat, amt]) => `${cat}: ₹${amt}`).join(', ')}
- Day patterns: ${Object.entries(dayOfWeekSpending).map(([day, amt]) => `${['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][Number(day)]}: ₹${amt}`).join(', ')}
- Monthly savings rate: ${profile.savings.rate.toFixed(1)}%

Identify:
1. Spending patterns (timing, categories, habits)
2. Areas where they could save
3. Specific recommendations

Respond in JSON:
{
  "patterns": [
    {"pattern": "Description of pattern", "frequency": "How often", "impact": "Financial impact"}
  ],
  "recommendations": ["Recommendation 1", "Recommendation 2"],
  "savingsOpportunities": [
    {"area": "Category/habit", "potentialSavings": 500, "suggestion": "How to save"}
  ]
}`;

  try {
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const text = response.text();
    
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      return JSON.parse(jsonMatch[0]);
    }
    
    return getDefaultPatternAnalysis(categoryTotals);
  } catch (error) {
    console.error('Gemini pattern analysis error:', error);
    return getDefaultPatternAnalysis(categoryTotals);
  }
}

/**
 * Generate personalized goal advice using Gemini
 */
export async function generateGoalAdvice(
  goal: { title: string; targetAmount: number; currentAmount: number; deadline?: Date },
  profile: FinancialProfile
): Promise<{
  status: string;
  advice: string;
  milestones: Array<{ milestone: string; amount: number; suggestedDate: string }>;
  tips: string[];
}> {
  const progress = (goal.currentAmount / goal.targetAmount) * 100;
  const remaining = goal.targetAmount - goal.currentAmount;
  const daysLeft = goal.deadline 
    ? Math.ceil((goal.deadline.getTime() - Date.now()) / (1000 * 60 * 60 * 24))
    : null;

  const prompt = `Help a gig worker in India achieve their financial goal.

GOAL: "${goal.title}"
- Target: ₹${goal.targetAmount}
- Current: ₹${goal.currentAmount} (${progress.toFixed(1)}% complete)
- Remaining: ₹${remaining}
${daysLeft ? `- Days until deadline: ${daysLeft}` : '- No specific deadline'}
${daysLeft ? `- Required per day: ₹${(remaining / daysLeft).toFixed(0)}` : ''}

USER CONTEXT:
- Monthly income: ~₹${profile.income.total}
- Current savings rate: ${profile.savings.rate.toFixed(1)}%
- Other active goals: ${profile.goals.active - 1}

Provide:
1. Goal status assessment
2. Personalized advice
3. Suggested milestones
4. Practical tips

Respond in JSON:
{
  "status": "On track / Needs attention / At risk - with brief explanation",
  "advice": "Personalized advice paragraph",
  "milestones": [
    {"milestone": "25% Complete", "amount": 2500, "suggestedDate": "2025-12-15"}
  ],
  "tips": ["Tip 1", "Tip 2", "Tip 3"]
}`;

  try {
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const text = response.text();
    
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      return JSON.parse(jsonMatch[0]);
    }
    
    return getDefaultGoalAdvice(goal, progress);
  } catch (error) {
    console.error('Gemini goal advice error:', error);
    return getDefaultGoalAdvice(goal, progress);
  }
}

// ============ Fallback Functions ============

function getDefaultAdvice(profile: FinancialProfile) {
  const insights = [];
  
  if (profile.savings.rate < 20) {
    insights.push({
      id: 'low_savings',
      type: 'warning' as const,
      category: 'savings',
      priority: 'high' as const,
      title: 'Boost Your Savings Rate',
      message: `Your savings rate is ${profile.savings.rate.toFixed(1)}%. Aim for at least 20% to build financial security.`,
      actionItems: ['Track all expenses', 'Find one area to cut back', 'Set up auto-transfer to savings']
    });
  } else {
    insights.push({
      id: 'good_savings',
      type: 'celebration' as const,
      category: 'savings',
      priority: 'low' as const,
      title: '🎉 Great Savings Rate!',
      message: `You're saving ${profile.savings.rate.toFixed(1)}% of your income. Excellent work!`,
      actionItems: ['Consider investing for growth', 'Set a stretch savings goal']
    });
  }

  if (profile.budgets.exceeded > 0) {
    insights.push({
      id: 'budget_exceeded',
      type: 'warning' as const,
      category: 'budgets',
      priority: 'high' as const,
      title: 'Budget Alert',
      message: `${profile.budgets.exceeded} budget(s) exceeded. Review your spending in these categories.`,
      actionItems: ['Check exceeded budgets', 'Adjust limits or spending', 'Set up alerts at 80%']
    });
  }

  return {
    insights,
    summary: `Based on your ${profile.savings.rate.toFixed(1)}% savings rate and current spending patterns, focus on maintaining consistent savings habits.`
  };
}

function getFallbackChatResponse(message: string, profile: FinancialProfile) {
  const lowerMessage = message.toLowerCase();
  let response = '';
  
  if (lowerMessage.includes('save') || lowerMessage.includes('saving')) {
    response = profile.savings.rate < 20
      ? `Your current savings rate is ${profile.savings.rate.toFixed(1)}%. Try the 50/30/20 rule: 50% needs, 30% wants, 20% savings.`
      : `Great job! You're saving ${profile.savings.rate.toFixed(1)}% of your income. Consider investing to grow your money.`;
  } else if (lowerMessage.includes('budget')) {
    response = profile.budgets.total === 0
      ? `You haven't set up any budgets yet. Start with your top expense category to gain control over spending.`
      : `You have ${profile.budgets.total} budget(s). ${profile.budgets.exceeded > 0 ? `${profile.budgets.exceeded} need attention.` : 'All on track!'}`;
  } else if (lowerMessage.includes('goal')) {
    response = profile.goals.active === 0
      ? `Setting financial goals gives purpose to your savings. Start with an emergency fund covering 3-6 months of expenses.`
      : `You have ${profile.goals.active} active goal(s). ${profile.goals.atRisk > 0 ? 'Some need attention!' : 'Keep going!'}`;
  } else {
    response = `I'm here to help with your finances! Ask me about saving tips, budgeting, goals, or spending analysis.`;
  }

  return {
    response,
    suggestions: ['How can I save more?', 'Analyze my spending', 'Help with budgets']
  };
}

function getDefaultWeeklySummary(weeklyData: { income: number; expenses: number; savings: number }) {
  const isPositive = weeklyData.savings > 0;
  return {
    headline: isPositive ? '💚 Positive Week!' : '⚠️ Watch Your Spending',
    highlights: [
      `Income: ₹${weeklyData.income}`,
      `Expenses: ₹${weeklyData.expenses}`,
      `${isPositive ? 'Saved' : 'Overspent'}: ₹${Math.abs(weeklyData.savings)}`
    ],
    tipOfTheWeek: 'Review your top expense category for potential savings.',
    encouragement: isPositive 
      ? 'Great job managing your money this week!' 
      : 'Every week is a fresh start. Let\'s do better next week!'
  };
}

function getDefaultPatternAnalysis(categoryTotals: { [key: string]: number }) {
  const topCategory = Object.entries(categoryTotals).sort((a, b) => b[1] - a[1])[0];
  return {
    patterns: [
      { pattern: `${topCategory?.[0] || 'Food'} is your highest spending category`, frequency: 'Regular', impact: 'Significant' }
    ],
    recommendations: [
      'Track daily expenses to identify small leaks',
      'Set category-specific budgets',
      'Review subscriptions monthly'
    ],
    savingsOpportunities: [
      { area: topCategory?.[0] || 'Food', potentialSavings: 500, suggestion: 'Look for ways to reduce by 10%' }
    ]
  };
}

function getDefaultGoalAdvice(goal: { title: string; targetAmount: number; currentAmount: number }, progress: number) {
  return {
    status: progress < 25 ? 'Just getting started' : progress < 75 ? 'Making progress' : 'Almost there!',
    advice: `Keep contributing regularly to reach your ${goal.title} goal.`,
    milestones: [
      { milestone: '25% Complete', amount: goal.targetAmount * 0.25, suggestedDate: 'In 1 month' },
      { milestone: '50% Complete', amount: goal.targetAmount * 0.5, suggestedDate: 'In 2 months' },
      { milestone: '75% Complete', amount: goal.targetAmount * 0.75, suggestedDate: 'In 3 months' }
    ],
    tips: [
      'Set up automatic transfers',
      'Celebrate small wins',
      'Visualize your goal to stay motivated'
    ]
  };
}

// Export FinancialProfile type for use in other files
export type { FinancialProfile };
