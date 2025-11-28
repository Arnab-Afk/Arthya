import express from 'express';
import {
  getTransactions,
  getTransaction,
  createTransaction,
  updateTransaction,
  deleteTransaction,
  getTransactionSummary,
  bulkImportTransactions,
  getSpendingByCategory,
  getMonthlyTrends,
} from '../controllers/transactionController';
import { protect } from '../middleware/auth';

const router = express.Router();

router.use(protect); // All routes are protected

router.route('/').get(getTransactions).post(createTransaction);
router.route('/summary').get(getTransactionSummary);
router.route('/bulk-import').post(bulkImportTransactions);
router.route('/by-category').get(getSpendingByCategory);
router.route('/monthly-trends').get(getMonthlyTrends);
router
  .route('/:id')
  .get(getTransaction)
  .put(updateTransaction)
  .delete(deleteTransaction);

export default router;
