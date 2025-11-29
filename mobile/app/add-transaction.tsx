/**
 * Add Transaction Screen
 * Manual entry form for income/expense transactions
 */

import React, { useState } from 'react';
import {
  StyleSheet,
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { useRouter, Stack } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import Animated, { FadeInDown } from 'react-native-reanimated';
import { Colors } from '@/constants/Colors';
import api from '@/services/api';

type TransactionType = 'income' | 'expense';

const EXPENSE_CATEGORIES = [
  { id: 'food', label: 'Food & Dining', icon: 'restaurant' },
  { id: 'transport', label: 'Transport', icon: 'car' },
  { id: 'shopping', label: 'Shopping', icon: 'cart' },
  { id: 'entertainment', label: 'Entertainment', icon: 'game-controller' },
  { id: 'utilities', label: 'Utilities', icon: 'flash' },
  { id: 'healthcare', label: 'Healthcare', icon: 'medkit' },
  { id: 'education', label: 'Education', icon: 'school' },
  { id: 'rent', label: 'Rent', icon: 'home' },
  { id: 'other', label: 'Other', icon: 'ellipsis-horizontal' },
];

const INCOME_CATEGORIES = [
  { id: 'salary', label: 'Salary', icon: 'briefcase' },
  { id: 'freelance', label: 'Freelance', icon: 'laptop' },
  { id: 'business', label: 'Business', icon: 'storefront' },
  { id: 'investment', label: 'Investment', icon: 'trending-up' },
  { id: 'gift', label: 'Gift', icon: 'gift' },
  { id: 'refund', label: 'Refund', icon: 'return-down-back' },
  { id: 'other', label: 'Other', icon: 'ellipsis-horizontal' },
];

export default function AddTransactionScreen() {
  const router = useRouter();
  const [type, setType] = useState<TransactionType>('expense');
  const [amount, setAmount] = useState('');
  const [category, setCategory] = useState('');
  const [description, setDescription] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const categories = type === 'expense' ? EXPENSE_CATEGORIES : INCOME_CATEGORIES;

  const handleSubmit = async () => {
    // Validation
    if (!amount || parseFloat(amount) <= 0) {
      Alert.alert('Error', 'Please enter a valid amount');
      return;
    }
    if (!category) {
      Alert.alert('Error', 'Please select a category');
      return;
    }

    setIsSubmitting(true);
    try {
      console.log('Creating transaction:', {
        type,
        amount: parseFloat(amount),
        category,
        description: description || `${type === 'income' ? 'Income' : 'Expense'} - ${category}`,
        date: new Date().toISOString(),
      });

      const response = await api.createTransaction({
        type,
        amount: parseFloat(amount),
        category,
        description: description || `${type === 'income' ? 'Income' : 'Expense'} - ${category}`,
        date: new Date().toISOString(),
      });

      console.log('Transaction response:', response);

      if (response.success) {
        Alert.alert(
          'Success',
          `${type === 'income' ? 'Income' : 'Expense'} of ₹${parseFloat(amount).toLocaleString('en-IN')} added successfully!`,
          [{ text: 'OK', onPress: () => router.replace('/(tabs)') }]
        );
      } else {
        throw new Error(response.message || 'Failed to add transaction');
      }
    } catch (error: any) {
      console.error('Transaction error:', error);
      Alert.alert('Error', error.message || 'Failed to add transaction. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <>
      <Stack.Screen
        options={{
          headerShown: false, // Hide the default Expo Router header
        }}
      />
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.container}
      >
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <Ionicons name="close" size={28} color={Colors.text} />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Add Transaction</Text>
          <View style={styles.backButton} />
        </View>

        <ScrollView
          contentContainerStyle={styles.content}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        >
          {/* Type Selector */}
          <Animated.View entering={FadeInDown.delay(100).duration(400)} style={styles.typeSelector}>
            <TouchableOpacity
              style={[styles.typeButton, type === 'expense' && styles.typeButtonActiveExpense]}
              onPress={() => {
                setType('expense');
                setCategory('');
              }}
            >
              <Ionicons
                name="arrow-down-circle"
                size={24}
                color={type === 'expense' ? '#fff' : Colors.expense}
              />
              <Text style={[styles.typeButtonText, type === 'expense' && styles.typeButtonTextActive]}>
                Expense
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.typeButton, type === 'income' && styles.typeButtonActiveIncome]}
              onPress={() => {
                setType('income');
                setCategory('');
              }}
            >
              <Ionicons
                name="arrow-up-circle"
                size={24}
                color={type === 'income' ? '#fff' : Colors.income}
              />
              <Text style={[styles.typeButtonText, type === 'income' && styles.typeButtonTextActive]}>
                Income
              </Text>
            </TouchableOpacity>
          </Animated.View>

          {/* Amount Input */}
          <Animated.View entering={FadeInDown.delay(200).duration(400)} style={styles.amountSection}>
            <Text style={styles.sectionLabel}>Amount</Text>
            <View style={styles.amountInputContainer}>
              <Text style={styles.currencySymbol}>₹</Text>
              <TextInput
                style={styles.amountInput}
                value={amount}
                onChangeText={setAmount}
                placeholder="0"
                placeholderTextColor={Colors.textDim}
                keyboardType="decimal-pad"
                autoFocus
              />
            </View>
          </Animated.View>

          {/* Category Selection */}
          <Animated.View entering={FadeInDown.delay(300).duration(400)} style={styles.categorySection}>
            <Text style={styles.sectionLabel}>Category</Text>
            <View style={styles.categoryGrid}>
              {categories.map((cat) => (
                <TouchableOpacity
                  key={cat.id}
                  style={[
                    styles.categoryItem,
                    category === cat.id && styles.categoryItemActive,
                  ]}
                  onPress={() => setCategory(cat.id)}
                >
                  <View
                    style={[
                      styles.categoryIconContainer,
                      category === cat.id && {
                        backgroundColor: type === 'expense' ? Colors.expense : Colors.income,
                      },
                    ]}
                  >
                    <Ionicons
                      name={cat.icon as any}
                      size={20}
                      color={category === cat.id ? '#fff' : Colors.text}
                    />
                  </View>
                  <Text
                    style={[
                      styles.categoryLabel,
                      category === cat.id && styles.categoryLabelActive,
                    ]}
                    numberOfLines={1}
                  >
                    {cat.label}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </Animated.View>

          {/* Description Input */}
          <Animated.View entering={FadeInDown.delay(400).duration(400)} style={styles.descriptionSection}>
            <Text style={styles.sectionLabel}>Description (Optional)</Text>
            <TextInput
              style={styles.descriptionInput}
              value={description}
              onChangeText={setDescription}
              placeholder="Add a note..."
              placeholderTextColor={Colors.textDim}
              multiline
              numberOfLines={3}
              maxLength={200}
            />
          </Animated.View>

          {/* Submit Button */}
          <Animated.View entering={FadeInDown.delay(500).duration(400)} style={styles.submitSection}>
            <TouchableOpacity
              style={[
                styles.submitButton,
                type === 'expense' ? styles.submitButtonExpense : styles.submitButtonIncome,
                isSubmitting && styles.submitButtonDisabled,
              ]}
              onPress={handleSubmit}
              disabled={isSubmitting}
            >
              {isSubmitting ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <>
                  <Ionicons name="checkmark-circle" size={24} color="#fff" />
                  <Text style={styles.submitButtonText}>
                    Add {type === 'expense' ? 'Expense' : 'Income'}
                  </Text>
                </>
              )}
            </TouchableOpacity>
          </Animated.View>
        </ScrollView>
      </KeyboardAvoidingView>
    </>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  backButton: {
    padding: 8,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: Colors.text,
  },
  content: {
    padding: 24,
    paddingBottom: 100,
  },
  typeSelector: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 32,
  },
  typeButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 16,
    borderRadius: 16,
    backgroundColor: Colors.card,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  typeButtonActiveExpense: {
    backgroundColor: Colors.expense,
    borderColor: Colors.expense,
  },
  typeButtonActiveIncome: {
    backgroundColor: Colors.income,
    borderColor: Colors.income,
  },
  typeButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: Colors.text,
  },
  typeButtonTextActive: {
    color: '#fff',
  },
  amountSection: {
    marginBottom: 32,
  },
  sectionLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: Colors.textDim,
    marginBottom: 12,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  amountInputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.card,
    borderRadius: 16,
    paddingHorizontal: 20,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  currencySymbol: {
    fontSize: 32,
    fontWeight: 'bold',
    color: Colors.primary,
    marginRight: 8,
  },
  amountInput: {
    flex: 1,
    fontSize: 40,
    fontWeight: 'bold',
    color: Colors.text,
    paddingVertical: 20,
  },
  categorySection: {
    marginBottom: 32,
  },
  categoryGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  categoryItem: {
    width: '30%',
    alignItems: 'center',
    padding: 12,
    borderRadius: 16,
    backgroundColor: Colors.card,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  categoryItemActive: {
    borderColor: Colors.primary,
    borderWidth: 2,
  },
  categoryIconContainer: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: Colors.cardElevated,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 8,
  },
  categoryLabel: {
    fontSize: 11,
    color: Colors.text,
    textAlign: 'center',
    fontWeight: '500',
  },
  categoryLabelActive: {
    color: Colors.primary,
    fontWeight: '600',
  },
  descriptionSection: {
    marginBottom: 32,
  },
  descriptionInput: {
    backgroundColor: Colors.card,
    borderRadius: 16,
    padding: 16,
    color: Colors.text,
    fontSize: 16,
    borderWidth: 1,
    borderColor: Colors.border,
    minHeight: 100,
    textAlignVertical: 'top',
  },
  submitSection: {
    marginTop: 16,
  },
  submitButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    paddingVertical: 18,
    borderRadius: 16,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 6,
  },
  submitButtonExpense: {
    backgroundColor: Colors.expense,
    shadowColor: Colors.expense,
  },
  submitButtonIncome: {
    backgroundColor: Colors.income,
    shadowColor: Colors.income,
  },
  submitButtonDisabled: {
    opacity: 0.6,
  },
  submitButtonText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
  },
});
