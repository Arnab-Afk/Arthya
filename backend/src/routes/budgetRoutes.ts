import express from 'express';
import { protect } from '../middleware/auth';
import {
  getBudgets,
  getBudget,
  createBudget,
  updateBudget,
  deleteBudget,
  getBudgetSummary,
} from '../controllers/budgetController';

const router = express.Router();

// All routes are protected
router.use(protect);

router.get('/summary', getBudgetSummary);
router.route('/').get(getBudgets).post(createBudget);
router.route('/:id').get(getBudget).put(updateBudget).delete(deleteBudget);

export default router;
