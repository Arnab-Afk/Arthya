import express from 'express';
import {
  getFinancialAdvice,
  getSpendingInsights,
  createSmartNotification,
  getAIInsights,
  getFinancialProfile,
  getWeeklySummary,
  getActionPlan,
  chatWithCoach,
  getGeminiAdvice,
  getGeminiWeeklySummary,
  getSpendingAnalysis,
  getGoalAdvice,
} from '../controllers/coachingController';
import { protect } from '../middleware/auth';

const router = express.Router();

router.use(protect); // All routes are protected

// Basic coaching
router.post('/advice', getFinancialAdvice);
router.get('/insights', getSpendingInsights);
router.post('/notify', createSmartNotification);

// AI-enhanced coaching (rule-based)
router.get('/ai-insights', getAIInsights);
router.get('/profile', getFinancialProfile);
router.get('/weekly-summary', getWeeklySummary);
router.get('/action-plan', getActionPlan);
router.post('/chat', chatWithCoach);

// Gemini AI-powered endpoints
router.get('/gemini-advice', getGeminiAdvice);
router.get('/gemini-weekly', getGeminiWeeklySummary);
router.get('/spending-analysis', getSpendingAnalysis);
router.get('/goal-advice/:goalId', getGoalAdvice);

export default router;
