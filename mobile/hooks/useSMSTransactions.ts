/**
 * useSMSTransactions Hook
 * Manages SMS reading and transaction parsing state
 */

import { useState, useCallback } from 'react';
import { Alert } from 'react-native';
import {
  isSMSReadingSupported,
  checkSMSPermission,
  requestSMSPermission,
  getTransactionsFromSMS,
  showPermissionDeniedAlert,
} from '../services/smsService';
import { ParsedTransaction } from '../services/smsParser';
import api from '../services/api';

interface UseSMSTransactionsState {
  isLoading: boolean;
  isSupported: boolean;
  hasPermission: boolean;
  transactions: ParsedTransaction[];
  stats: {
    totalSMS: number;
    transactionSMS: number;
    imported: number;
  };
  error: string | null;
}

interface UseSMSTransactionsActions {
  checkPermission: () => Promise<boolean>;
  requestPermission: () => Promise<boolean>;
  scanSMS: (daysBack?: number) => Promise<void>;
  importTransactions: (transactions: ParsedTransaction[]) => Promise<number>;
  importAll: () => Promise<void>;
  clearTransactions: () => void;
}

export function useSMSTransactions(): UseSMSTransactionsState & UseSMSTransactionsActions {
  const [isLoading, setIsLoading] = useState(false);
  const [hasPermission, setHasPermission] = useState(false);
  const [transactions, setTransactions] = useState<ParsedTransaction[]>([]);
  const [stats, setStats] = useState({ totalSMS: 0, transactionSMS: 0, imported: 0 });
  const [error, setError] = useState<string | null>(null);

  const isSupported = isSMSReadingSupported();

  const checkPermission = useCallback(async (): Promise<boolean> => {
    if (!isSupported) return false;
    const result = await checkSMSPermission();
    setHasPermission(result);
    return result;
  }, [isSupported]);

  const requestPermission = useCallback(async (): Promise<boolean> => {
    if (!isSupported) {
      Alert.alert('Not Supported', 'SMS reading is only available on Android devices.');
      return false;
    }

    const result = await requestSMSPermission();
    setHasPermission(result.granted);

    if (!result.granted) {
      showPermissionDeniedAlert();
    }

    return result.granted;
  }, [isSupported]);

  const scanSMS = useCallback(async (daysBack = 30): Promise<void> => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await getTransactionsFromSMS({ daysBack });

      if (!result.success) {
        setError(result.error || 'Failed to read SMS');
        return;
      }

      setTransactions(result.transactions);
      setStats({
        totalSMS: result.totalSMS,
        transactionSMS: result.transactionSMS,
        imported: 0,
      });
    } catch (err: any) {
      setError(err.message || 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const importTransactions = useCallback(async (
    txnsToImport: ParsedTransaction[]
  ): Promise<number> => {
    let importedCount = 0;

    for (const txn of txnsToImport) {
      try {
        await api.createTransaction({
          type: txn.type,
          category: txn.category,
          amount: txn.amount,
          description: txn.description,
          date: txn.date.toISOString(),
          recipient: txn.recipient,
        });
        importedCount++;
      } catch (err) {
        console.error('Failed to import transaction:', err);
      }
    }

    setStats(prev => ({ ...prev, imported: prev.imported + importedCount }));
    return importedCount;
  }, []);

  const importAll = useCallback(async (): Promise<void> => {
    if (transactions.length === 0) {
      Alert.alert('No Transactions', 'Please scan SMS first to find transactions.');
      return;
    }

    setIsLoading(true);

    try {
      const count = await importTransactions(transactions);
      Alert.alert(
        'Import Complete',
        `Successfully imported ${count} of ${transactions.length} transactions.`
      );
      setTransactions([]);
    } catch (err: any) {
      Alert.alert('Import Failed', err.message || 'Failed to import transactions');
    } finally {
      setIsLoading(false);
    }
  }, [transactions, importTransactions]);

  const clearTransactions = useCallback(() => {
    setTransactions([]);
    setStats({ totalSMS: 0, transactionSMS: 0, imported: 0 });
    setError(null);
  }, []);

  return {
    isLoading,
    isSupported,
    hasPermission,
    transactions,
    stats,
    error,
    checkPermission,
    requestPermission,
    scanSMS,
    importTransactions,
    importAll,
    clearTransactions,
  };
}
