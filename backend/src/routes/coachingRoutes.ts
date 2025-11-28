import express from 'express';
import {
  getFinancialAdvice,
  getSpendingInsights,
  createSmartNotification,
} from '../controllers/coachingController';
import { protect } from '../middleware/auth';

const router = express.Router();

router.use(protect); // All routes are protected

router.post('/advice', getFinancialAdvice);
router.get('/insights', getSpendingInsights);
router.post('/notify', createSmartNotification);

export default router;
