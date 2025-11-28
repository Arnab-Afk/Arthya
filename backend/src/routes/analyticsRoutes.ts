import express from 'express';
import {
  getDashboard,
  getSpendingTrends,
  getIncomeAnalysis,
} from '../controllers/analyticsController';
import { protect } from '../middleware/auth';

const router = express.Router();

router.use(protect); // All routes are protected

router.get('/dashboard', getDashboard);
router.get('/trends', getSpendingTrends);
router.get('/income', getIncomeAnalysis);

export default router;
