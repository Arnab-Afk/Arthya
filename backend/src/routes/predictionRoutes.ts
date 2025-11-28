import express from 'express';
import { protect } from '../middleware/auth';
import {
  getIncomePrediction,
  getExpensePrediction,
  getCashFlowProjection,
  getNudges,
  getFinancialHealthScore,
} from '../controllers/predictionController';

const router = express.Router();

// All routes are protected
router.use(protect);

router.get('/income', getIncomePrediction);
router.get('/expenses', getExpensePrediction);
router.get('/cashflow', getCashFlowProjection);
router.get('/nudges', getNudges);
router.get('/health-score', getFinancialHealthScore);

export default router;
