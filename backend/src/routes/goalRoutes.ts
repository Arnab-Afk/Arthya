import express from 'express';
import {
  getGoals,
  getGoal,
  createGoal,
  updateGoal,
  updateGoalProgress,
  deleteGoal,
} from '../controllers/goalController';
import { protect } from '../middleware/auth';

const router = express.Router();

router.use(protect); // All routes are protected

router.route('/').get(getGoals).post(createGoal);
router
  .route('/:id')
  .get(getGoal)
  .put(updateGoal)
  .delete(deleteGoal);
router.route('/:id/progress').put(updateGoalProgress);

export default router;
