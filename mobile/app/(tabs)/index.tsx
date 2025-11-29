import React, { useState, useEffect } from 'react';
import { StyleSheet, View, Text, ScrollView, TouchableOpacity, Dimensions, Platform, RefreshControl, ActivityIndicator } from 'react-native';
import { Colors } from '@/constants/Colors';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { SummaryCard } from '@/components/SummaryCard';
import { GoalCard } from '@/components/GoalCard';
import Animated, { FadeInDown, FadeInRight } from 'react-native-reanimated';
import { useAuth } from '@/contexts/AuthContext';
import api from '@/services/api';
import { DashboardData, Transaction, Goal } from '@/types/api';
import { useRouter } from 'expo-router';

const { width } = Dimensions.get('window');

export default function HomeScreen() {
  const { user } = useAuth();
  const router = useRouter();
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [goals, setGoals] = useState<Goal[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadDashboard();
  }, []);

  const loadDashboard = async () => {
    try {
      const [dashboardRes, goalsRes] = await Promise.all([
        api.getDashboard(),
        api.getGoals()
      ]);
      
      if (dashboardRes.success && dashboardRes.data) {
        setDashboard(dashboardRes.data);
      }
      if (goalsRes.success && goalsRes.data) {
        setGoals(goalsRes.data.filter((g: Goal) => g.status === 'active'));
      }
    } catch (error) {
      console.error('Failed to load dashboard:', error);
    } finally {
      setLoading(false);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadDashboard();
    setRefreshing(false);
  };

  if (loading) {
    return (
      <View style={[styles.container, styles.centerContent]}>
        <ActivityIndicator size="large" color={Colors.primary} />
        <Text style={styles.loadingText}>Loading your dashboard...</Text>
      </View>
    );
  }

  const formatCurrency = (amount: number | string) => {
    const num = typeof amount === 'string' ? parseFloat(amount) : amount;
    return `₹${num.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  const getInitials = (name?: string) => {
    if (!name) return 'U';
    return name.split(' ').map(n => n[0]).join('').toUpperCase().substring(0, 2);
  };

  const income = dashboard?.summary?.totalIncome || 0;
  const expense = dashboard?.summary?.totalExpense || 0;
  const balance = dashboard?.summary?.balance || (income - expense);
  const savingsRate = dashboard?.summary?.savingsRate || 0;

  return (
    <View style={styles.container}>
      {/* Header */}
      <Animated.View entering={FadeInDown.delay(100).duration(500)} style={styles.header}>
        <TouchableOpacity onPress={() => router.push('/chat')}>
          <Ionicons name="chatbubble-ellipses-outline" size={24} color={Colors.text} />
        </TouchableOpacity>
        <View style={styles.headerRight}>
          <TouchableOpacity style={styles.iconButton}>
            <Ionicons name="notifications-outline" size={24} color={Colors.text} />
            <View style={styles.badge} />
          </TouchableOpacity>
          <TouchableOpacity style={styles.profileButton} onPress={() => router.push('/(tabs)/explore')}>
            <Text style={styles.profileText}>{getInitials(user?.name)}</Text>
          </TouchableOpacity>
        </View>
      </Animated.View>

      <ScrollView 
        contentContainerStyle={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={Colors.primary} />
        }
      >
        {/* Balance Card */}
        <Animated.View entering={FadeInRight.delay(200).duration(500)}>
          <LinearGradient
            colors={['#333', '#111']}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={styles.card}
          >
            <View style={styles.cardHeader}>
              <Ionicons name="wallet" size={24} color={Colors.primary} />
              <Text style={styles.cardTitle}>Total Balance</Text>
            </View>
            <Text style={styles.balanceAmount}>{formatCurrency(balance)}</Text>
            <View style={styles.cardFooter}>
              <Text style={styles.cardLabel}>
                Savings Rate: {savingsRate.toFixed(1)}%
              </Text>
            </View>
          </LinearGradient>
        </Animated.View>

        {/* Summary Cards */}
        <Animated.View entering={FadeInDown.delay(300).duration(500)} style={styles.summarySection}>
          <SummaryCard 
            title="Income" 
            amount={formatCurrency(income)} 
            icon="arrow-down" 
            color="#9FE8AE" 
          />
          <SummaryCard 
            title="Expense" 
            amount={formatCurrency(expense)} 
            icon="arrow-up" 
            color="#FF453A" 
          />
          <SummaryCard 
            title="Savings" 
            amount={formatCurrency(income - expense)} 
            icon="wallet" 
            color="#D4B483" 
          />
        </Animated.View>

        {/* Goals & Milestones */}
        <Animated.View entering={FadeInDown.delay(400).duration(500)} style={styles.goalsSection}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Goals & Milestones</Text>
            <TouchableOpacity>
              <Text style={styles.seeAllText}>See All</Text>
            </TouchableOpacity>
          </View>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.goalsScroll}>
            {goals.length > 0 ? (
              goals.slice(0, 3).map((goal: Goal) => (
                <GoalCard
                  key={goal.id}
                  title={goal.name}
                  currentAmount={formatCurrency(goal.currentAmount)}
                  targetAmount={formatCurrency(goal.targetAmount)}
                  progress={goal.progress / 100}
                  icon="flag"
                />
              ))
            ) : (
              <View style={styles.emptyGoals}>
                <Ionicons name="flag-outline" size={48} color={Colors.textDim} />
                <Text style={styles.emptyText}>No active goals</Text>
                <Text style={styles.emptySubtext}>Create your first financial goal</Text>
              </View>
            )}
          </ScrollView>
        </Animated.View>

        {/* Quick Actions */}
        <Animated.View entering={FadeInDown.delay(450).duration(500)} style={styles.quickActionsSection}>
          <Text style={styles.sectionTitle}>Quick Actions</Text>
          <View style={styles.quickActionsRow}>
            <TouchableOpacity 
              style={styles.quickActionButton} 
              onPress={() => router.push('/sms-import')}
            >
              <View style={[styles.quickActionIcon, { backgroundColor: '#9FE8AE22' }]}>
                <Ionicons name="mail" size={24} color="#9FE8AE" />
              </View>
              <Text style={styles.quickActionText}>Import SMS</Text>
            </TouchableOpacity>
            <TouchableOpacity 
              style={styles.quickActionButton}
              onPress={() => router.push('/chat')}
            >
              <View style={[styles.quickActionIcon, { backgroundColor: '#64D2FF22' }]}>
                <Ionicons name="chatbubble-ellipses" size={24} color="#64D2FF" />
              </View>
              <Text style={styles.quickActionText}>AI Coach</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.quickActionButton}>
              <View style={[styles.quickActionIcon, { backgroundColor: '#D4B48322' }]}>
                <Ionicons name="flag" size={24} color="#D4B483" />
              </View>
              <Text style={styles.quickActionText}>New Goal</Text>
            </TouchableOpacity>
          </View>
        </Animated.View>

        {/* Recent Transactions */}
        <Animated.View entering={FadeInDown.delay(500).duration(500)} style={styles.transferSection}>
          <Text style={styles.sectionTitle}>Recent Transactions</Text>
          {dashboard?.recentTransactions && dashboard.recentTransactions.length > 0 ? (
            <View style={styles.transactionsList}>
              {dashboard.recentTransactions.slice(0, 5).map((transaction: Transaction) => (
                <View key={transaction.id} style={styles.transactionItem}>
                  <View style={styles.transactionLeft}>
                    <View style={[
                      styles.transactionIcon,
                      { backgroundColor: transaction.type === 'income' ? '#9FE8AE22' : '#FF453A22' }
                    ]}>
                      <Ionicons 
                        name={transaction.type === 'income' ? 'arrow-down' : 'arrow-up'} 
                        size={20} 
                        color={transaction.type === 'income' ? '#9FE8AE' : '#FF453A'} 
                      />
                    </View>
                    <View>
                      <Text style={styles.transactionCategory}>{transaction.category}</Text>
                      <Text style={styles.transactionDate}>
                        {new Date(transaction.date).toLocaleDateString()}
                      </Text>
                    </View>
                  </View>
                  <Text style={[
                    styles.transactionAmount,
                    { color: transaction.type === 'income' ? '#9FE8AE' : '#FF453A' }
                  ]}>
                    {transaction.type === 'income' ? '+' : '-'}{formatCurrency(transaction.amount)}
                  </Text>
                </View>
              ))}
            </View>
          ) : (
            <View style={styles.emptyTransactions}>
              <Ionicons name="receipt-outline" size={48} color={Colors.textDim} />
              <Text style={styles.emptyText}>No transactions yet</Text>
              <Text style={styles.emptySubtext}>Start tracking your finances</Text>
            </View>
          )}
        </Animated.View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
    paddingTop: 60,
  },
  centerContent: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: Colors.textDim,
    marginTop: 16,
    fontSize: 14,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 24,
    marginBottom: 24,
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  iconButton: {
    marginLeft: 20,
  },
  badge: {
    position: 'absolute',
    top: -2,
    right: -2,
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: Colors.accent,
    borderWidth: 1,
    borderColor: Colors.background,
  },
  profileButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: Colors.card,
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 20,
    borderWidth: 2,
    borderColor: Colors.primary,
    shadowColor: Colors.primary,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 4,
  },
  profileText: {
    color: '#fff',
    fontWeight: 'bold',
  },
  content: {
    paddingBottom: 100,
    paddingHorizontal: 24,
  },
  card: {
    width: '100%',
    borderRadius: 24,
    padding: 24,
    marginBottom: 24,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.4,
    shadowRadius: 12,
    elevation: 10,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  cardTitle: {
    color: Colors.text,
    fontSize: 16,
    fontWeight: '600',
  },
  balanceAmount: {
    color: Colors.primary,
    fontSize: 36,
    fontWeight: 'bold',
    marginVertical: 16,
  },
  cardFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  cardLabel: {
    color: '#AAAAAA',
    fontSize: 12,
  },
  transferSection: {
    marginBottom: 32,
  },
  sectionTitle: {
    color: Colors.text,
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 16,
  },
  transferScroll: {
    marginLeft: -8,
  },
  userCard: {
    backgroundColor: Colors.card,
    width: 80,
    height: 120,
    borderRadius: 40,
    alignItems: 'center',
    justifyContent: 'center',
    marginLeft: 8,
    paddingVertical: 10,
    borderWidth: 1,
    borderColor: Colors.border,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.12,
    shadowRadius: 5,
    elevation: 3,
  },
  userImage: {
    width: 50,
    height: 50,
    borderRadius: 25,
    marginBottom: 12,
  },
  userName: {
    color: Colors.text,
    fontSize: 13,
    fontWeight: '500',
  },
  userId: {
    color: Colors.textDim,
    fontSize: 11,
    marginTop: 4,
  },
  summarySection: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 24,
    marginBottom: 32,
  },
  goalsSection: {
    marginBottom: 32,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 24,
    marginBottom: 16,
  },
  seeAllText: {
    color: Colors.primary,
    fontSize: 14,
    fontWeight: '600',
  },
  goalsScroll: {
    paddingLeft: 24,
  },
  emptyGoals: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 40,
    marginHorizontal: 24,
    backgroundColor: Colors.card,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  emptyTransactions: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 40,
    backgroundColor: Colors.card,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  emptyText: {
    color: Colors.text,
    fontSize: 16,
    fontWeight: '600',
    marginTop: 16,
  },
  emptySubtext: {
    color: Colors.textDim,
    fontSize: 14,
    marginTop: 4,
  },
  transactionsList: {
    gap: 12,
  },
  transactionItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    backgroundColor: Colors.card,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  transactionLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  transactionIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
  },
  transactionCategory: {
    color: Colors.text,
    fontSize: 15,
    fontWeight: '600',
  },
  transactionDate: {
    color: Colors.textDim,
    fontSize: 12,
    marginTop: 2,
  },
  transactionAmount: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  quickActionsSection: {
    paddingHorizontal: 24,
    marginBottom: 24,
  },
  quickActionsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  quickActionButton: {
    alignItems: 'center',
    width: '30%',
  },
  quickActionIcon: {
    width: 56,
    height: 56,
    borderRadius: 28,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 8,
  },
  quickActionText: {
    color: Colors.textDim,
    fontSize: 12,
    textAlign: 'center',
  },
});
