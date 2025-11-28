/**
 * SMS Import Component
 * UI for scanning and importing transactions from SMS
 */

import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  FlatList,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Colors } from '@/constants/Colors';
import { useSMSTransactions } from '@/hooks/useSMSTransactions';
import { ParsedTransaction, getCategoryIcon } from '@/services/smsParser';

interface SMSImportProps {
  onImportComplete?: (count: number) => void;
  onClose?: () => void;
}

export function SMSImportModal({ onImportComplete, onClose }: SMSImportProps) {
  const {
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
    clearTransactions,
  } = useSMSTransactions();

  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [daysBack, setDaysBack] = useState(30);

  useEffect(() => {
    checkPermission();
  }, [checkPermission]);

  const handleScan = async () => {
    if (!hasPermission) {
      const granted = await requestPermission();
      if (!granted) return;
    }
    await scanSMS(daysBack);
    // Select all by default
    setSelectedIds(new Set(transactions.map((_, i) => i)));
  };

  useEffect(() => {
    if (transactions.length > 0) {
      setSelectedIds(new Set(transactions.map((_, i) => i)));
    }
  }, [transactions]);

  const toggleSelection = (index: number) => {
    const newSet = new Set(selectedIds);
    if (newSet.has(index)) {
      newSet.delete(index);
    } else {
      newSet.add(index);
    }
    setSelectedIds(newSet);
  };

  const selectAll = () => {
    setSelectedIds(new Set(transactions.map((_, i) => i)));
  };

  const deselectAll = () => {
    setSelectedIds(new Set());
  };

  const handleImport = async () => {
    const selected = transactions.filter((_, i) => selectedIds.has(i));
    if (selected.length === 0) {
      Alert.alert('No Selection', 'Please select transactions to import.');
      return;
    }

    Alert.alert(
      'Import Transactions',
      `Import ${selected.length} transactions?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Import',
          onPress: async () => {
            const count = await importTransactions(selected);
            onImportComplete?.(count);
            Alert.alert('Success', `Imported ${count} transactions!`);
            clearTransactions();
          },
        },
      ]
    );
  };

  const formatDate = (date: Date) => {
    return date.toLocaleDateString('en-IN', {
      day: 'numeric',
      month: 'short',
      year: 'numeric',
    });
  };

  const formatAmount = (amount: number, type: string) => {
    const prefix = type === 'income' ? '+' : '-';
    return `${prefix}₹${amount.toLocaleString('en-IN')}`;
  };

  const renderTransaction = ({ item, index }: { item: ParsedTransaction; index: number }) => {
    const isSelected = selectedIds.has(index);
    const icon = getCategoryIcon(item.category);

    return (
      <TouchableOpacity
        style={[styles.transactionItem, isSelected && styles.transactionItemSelected]}
        onPress={() => toggleSelection(index)}
      >
        <View style={styles.checkbox}>
          <Ionicons
            name={isSelected ? 'checkbox' : 'square-outline'}
            size={24}
            color={isSelected ? Colors.primary : Colors.textDim}
          />
        </View>
        <View style={styles.iconContainer}>
          <Text style={styles.categoryIcon}>{icon}</Text>
        </View>
        <View style={styles.transactionDetails}>
          <Text style={styles.transactionCategory}>
            {item.category.replace(/_/g, ' ').toUpperCase()}
          </Text>
          <Text style={styles.transactionDescription} numberOfLines={1}>
            {item.description}
          </Text>
          <Text style={styles.transactionDate}>{formatDate(item.date)}</Text>
        </View>
        <Text
          style={[
            styles.transactionAmount,
            item.type === 'income' ? styles.incomeAmount : styles.expenseAmount,
          ]}
        >
          {formatAmount(item.amount, item.type)}
        </Text>
      </TouchableOpacity>
    );
  };

  if (!isSupported) {
    return (
      <View style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.title}>Import from SMS</Text>
          <TouchableOpacity onPress={onClose} style={styles.closeButton}>
            <Ionicons name="close" size={24} color={Colors.text} />
          </TouchableOpacity>
        </View>
        <View style={styles.centerContent}>
          <Ionicons name="phone-portrait-outline" size={64} color={Colors.textDim} />
          <Text style={styles.unsupportedText}>
            SMS import is only available on Android devices.
          </Text>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Import from SMS</Text>
        <TouchableOpacity onPress={onClose} style={styles.closeButton}>
          <Ionicons name="close" size={24} color={Colors.text} />
        </TouchableOpacity>
      </View>

      {/* Days selector */}
      <View style={styles.daysSelector}>
        <Text style={styles.daysLabel}>Scan last:</Text>
        {[7, 30, 90].map((days) => (
          <TouchableOpacity
            key={days}
            style={[styles.daysButton, daysBack === days && styles.daysButtonActive]}
            onPress={() => setDaysBack(days)}
          >
            <Text
              style={[
                styles.daysButtonText,
                daysBack === days && styles.daysButtonTextActive,
              ]}
            >
              {days} days
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Scan button */}
      <TouchableOpacity
        style={styles.scanButton}
        onPress={handleScan}
        disabled={isLoading}
      >
        {isLoading ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <>
            <Ionicons name="scan" size={20} color="#fff" />
            <Text style={styles.scanButtonText}>Scan SMS Messages</Text>
          </>
        )}
      </TouchableOpacity>

      {/* Stats */}
      {stats.transactionSMS > 0 && (
        <View style={styles.statsContainer}>
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{stats.totalSMS}</Text>
            <Text style={styles.statLabel}>Total SMS</Text>
          </View>
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{stats.transactionSMS}</Text>
            <Text style={styles.statLabel}>Bank SMS</Text>
          </View>
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{transactions.length}</Text>
            <Text style={styles.statLabel}>Transactions</Text>
          </View>
        </View>
      )}

      {/* Error */}
      {error && (
        <View style={styles.errorContainer}>
          <Ionicons name="warning" size={20} color={Colors.expense} />
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}

      {/* Transaction list */}
      {transactions.length > 0 && (
        <>
          <View style={styles.selectionHeader}>
            <Text style={styles.selectionText}>
              {selectedIds.size} of {transactions.length} selected
            </Text>
            <View style={styles.selectionButtons}>
              <TouchableOpacity onPress={selectAll} style={styles.selectionButton}>
                <Text style={styles.selectionButtonText}>Select All</Text>
              </TouchableOpacity>
              <TouchableOpacity onPress={deselectAll} style={styles.selectionButton}>
                <Text style={styles.selectionButtonText}>Clear</Text>
              </TouchableOpacity>
            </View>
          </View>

          <FlatList
            data={transactions}
            renderItem={renderTransaction}
            keyExtractor={(_, index) => index.toString()}
            style={styles.list}
            showsVerticalScrollIndicator={false}
          />

          <TouchableOpacity
            style={[styles.importButton, selectedIds.size === 0 && styles.importButtonDisabled]}
            onPress={handleImport}
            disabled={selectedIds.size === 0 || isLoading}
          >
            <Ionicons name="download" size={20} color="#fff" />
            <Text style={styles.importButtonText}>
              Import {selectedIds.size} Transactions
            </Text>
          </TouchableOpacity>
        </>
      )}

      {/* Empty state */}
      {!isLoading && transactions.length === 0 && stats.totalSMS === 0 && (
        <View style={styles.emptyState}>
          <Ionicons name="mail-outline" size={64} color={Colors.textDim} />
          <Text style={styles.emptyStateText}>
            Tap &quot;Scan SMS Messages&quot; to find transactions in your bank SMS.
          </Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
    padding: 16,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
    paddingTop: 8,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: Colors.text,
  },
  closeButton: {
    padding: 8,
  },
  daysSelector: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  daysLabel: {
    color: Colors.textDim,
    marginRight: 12,
  },
  daysButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    backgroundColor: Colors.card,
    marginRight: 8,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  daysButtonActive: {
    backgroundColor: Colors.primary,
    borderColor: Colors.primary,
  },
  daysButtonText: {
    color: Colors.textDim,
    fontSize: 12,
  },
  daysButtonTextActive: {
    color: '#fff',
    fontWeight: '600',
  },
  scanButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: Colors.primary,
    paddingVertical: 14,
    borderRadius: 12,
    gap: 8,
  },
  scanButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  statsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    backgroundColor: Colors.card,
    borderRadius: 12,
    padding: 16,
    marginTop: 16,
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: Colors.text,
  },
  statLabel: {
    fontSize: 12,
    color: Colors.textDim,
    marginTop: 4,
  },
  errorContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    padding: 12,
    borderRadius: 8,
    marginTop: 16,
    gap: 8,
  },
  errorText: {
    color: Colors.expense,
    flex: 1,
  },
  selectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 16,
    marginBottom: 8,
  },
  selectionText: {
    color: Colors.textDim,
  },
  selectionButtons: {
    flexDirection: 'row',
    gap: 12,
  },
  selectionButton: {
    padding: 4,
  },
  selectionButtonText: {
    color: Colors.primary,
    fontSize: 14,
  },
  list: {
    flex: 1,
    marginTop: 8,
  },
  transactionItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.card,
    borderRadius: 12,
    padding: 12,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  transactionItemSelected: {
    borderColor: Colors.primary,
    backgroundColor: 'rgba(59, 130, 246, 0.1)',
  },
  checkbox: {
    marginRight: 12,
  },
  iconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: Colors.background,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  categoryIcon: {
    fontSize: 20,
  },
  transactionDetails: {
    flex: 1,
  },
  transactionCategory: {
    fontSize: 12,
    color: Colors.textDim,
    fontWeight: '600',
  },
  transactionDescription: {
    fontSize: 14,
    color: Colors.text,
    marginTop: 2,
  },
  transactionDate: {
    fontSize: 12,
    color: Colors.textDim,
    marginTop: 2,
  },
  transactionAmount: {
    fontSize: 16,
    fontWeight: '600',
  },
  incomeAmount: {
    color: Colors.income,
  },
  expenseAmount: {
    color: Colors.expense,
  },
  importButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: Colors.income,
    paddingVertical: 14,
    borderRadius: 12,
    gap: 8,
    marginTop: 16,
    marginBottom: 16,
  },
  importButtonDisabled: {
    opacity: 0.5,
  },
  importButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  centerContent: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 32,
  },
  unsupportedText: {
    color: Colors.textDim,
    textAlign: 'center',
    marginTop: 16,
    fontSize: 16,
  },
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 32,
  },
  emptyStateText: {
    color: Colors.textDim,
    textAlign: 'center',
    marginTop: 16,
    fontSize: 14,
    lineHeight: 22,
  },
});
