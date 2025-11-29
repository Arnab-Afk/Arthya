import React, { useState, useEffect, useCallback } from 'react';
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
import { useFocusEffect } from '@react-navigation/native';

const { width } = Dimensions.get('window');

export default function HomeScreen() {
  const { user } = useAuth();
  const router = useRouter();
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [goals, setGoals] = useState<Goal[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Refresh data when screen comes into focus
  useFocusEffect(
    useCallback(() => {
      console.log('Home screen focused - refreshing data');
      loadDashboard();
    }, [])
  );

  const loadDashboard = async () => {
    try {
      console.log('Loading dashboard data...');
      const [dashboardRes, goalsRes] = await Promise.all([
        api.getDashboard(),
        api.getGoals()
      ]);
      
      console.log('Dashboard response:', JSON.stringify(dashboardRes, null, 2));
      
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

  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good Morning';
    if (hour < 18) return 'Good Afternoon';
    return 'Good Evening';
  };

  // Backend returns 'income' and 'expense', not 'totalIncome' and 'totalExpense'
  const income = dashboard?.summary?.totalIncome || dashboard?.summary?.income || 0;
  const expense = dashboard?.summary?.totalExpense || dashboard?.summary?.expense || 0;
  const balance = dashboard?.summary?.balance || dashboard?.summary?.savings || (income - expense);
  const savingsRate = dashboard?.summary?.savingsRate || (income > 0 ? Math.round(((income - expense) / income) * 100) : 0);

  return (
    <View style={styles.container}>
      {/* Header */}
      <Animated.View entering={FadeInDown.delay(100).duration(500)} style={styles.header}>
        <View>
          <Text style={styles.greetingText}>{getGreeting()},</Text>
          <Text style={styles.userNameText}>{user?.name?.split(' ')[0] || 'User'}</Text>
        </View>
        <View style={styles.headerRight}>
          <TouchableOpacity style={styles.iconButton} onPress={() => router.push('/chat')}>
            <Ionicons name="chatbubble-ellipses-outline" size={24} color={Colors.text} />
          </TouchableOpacity>
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
        showsVerticalScrollIndicator={false}
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
              <View style={styles.walletIconContainer}>
                <Ionicons name="wallet" size={20} color="#fff" />
              </View>
              <Text style={styles.cardTitle}>Total Balance</Text>
            </View>
            <Text style={styles.balanceAmount}>{formatCurrency(balance)}</Text>
            <View style={styles.cardFooter}>
              <View style={styles.savingsBadge}>
                <Ionicons name="trending-up" size={14} color={Colors.primary} />
                <Text style={styles.savingsText}>
                  {savingsRate.toFixed(1)}% Savings Rate
                </Text>
              </View>
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

        {/* Quick Actions */}
        <Animated.View entering={FadeInDown.delay(400).duration(500)} style={styles.quickActionsSection}>
          <Text style={styles.sectionTitle}>Quick Actions</Text>
          <View style={styles.quickActionsRow}>
            <TouchableOpacity 
              style={styles.quickActionButton} 
              onPress={() => router.push('/sms-import')}
            >
              <LinearGradient
                colors={['rgba(159, 232, 174, 0.1)', 'rgba(159, 232, 174, 0.05)']}
                style={styles.quickActionGradient}
              >
                <Ionicons name="mail" size={24} color="#9FE8AE" />
              </LinearGradient>
              <Text style={styles.quickActionText}>Import SMS</Text>
            </TouchableOpacity>
            
            <TouchableOpacity 
              style={styles.quickActionButton}
              onPress={() => router.push('/chat')}
            >
              <LinearGradient
                colors={['rgba(100, 210, 255, 0.1)', 'rgba(100, 210, 255, 0.05)']}
                style={styles.quickActionGradient}
              >
                <Ionicons name="chatbubble-ellipses" size={24} color="#64D2FF" />
              </LinearGradient>
              <Text style={styles.quickActionText}>AI Coach</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.quickActionButton}>
              <LinearGradient
                colors={['rgba(212, 180, 131, 0.1)', 'rgba(212, 180, 131, 0.05)']}
                style={styles.quickActionGradient}
              >
                <Ionicons name="flag" size={24} color="#D4B483" />
              </LinearGradient>
              <Text style={styles.quickActionText}>New Goal</Text>
            </TouchableOpacity>
          </View>
        </Animated.View>

        {/* Goals & Milestones */}
        <Animated.View entering={FadeInDown.delay(450).duration(500)} style={styles.goalsSection}>
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
                      { backgroundColor: transaction.type === 'income' ? 'rgba(159, 232, 174, 0.1)' : 'rgba(255, 69, 58, 0.1)' }
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

      {/* Floating Action Button */}
      <TouchableOpacity
        style={styles.fab}
        onPress={() => router.push('/add-transaction')}
        activeOpacity={0.8}
      >
        <Ionicons name="add" size={32} color="#000" />
      </TouchableOpacity>
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
  greetingText: {
    color: Colors.textDim,
    fontSize: 14,
    fontWeight: '500',
  },
  userNameText: {
    color: Colors.text,
    fontSize: 24,
    fontWeight: 'bold',
    marginTop: 4,
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  iconButton: {
    marginLeft: 16,
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: Colors.card,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: Colors.border,
  },
  badge: {
    position: 'absolute',
    top: 8,
    right: 8,
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: Colors.accent,
    borderWidth: 1,
    borderColor: Colors.card,
  },
  profileButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: Colors.primary,
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 16,
    shadowColor: Colors.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 4,
  },
  profileText: {
    color: '#000',
    fontWeight: 'bold',
    fontSize: 16,
  },
  content: {
    paddingBottom: 100,
    paddingHorizontal: 24,
  },
  card: {
    width: '100%',
    borderRadius: 32,
    padding: 24,
    marginBottom: 24,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.05)',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.5,
    shadowRadius: 20,
    elevation: 10,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 8,
  },
  walletIconContainer: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: 'rgba(255,255,255,0.1)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  cardTitle: {
    color: Colors.textDim,
    fontSize: 16,
    fontWeight: '500',
  },
  balanceAmount: {
    color: '#fff',
    fontSize: 42,
    fontWeight: 'bold',
    marginVertical: 12,
    letterSpacing: -1,
  },
  cardFooter: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  savingsBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(159, 232, 174, 0.1)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    gap: 6,
  },
  savingsText: {
    color: Colors.primary,
    fontSize: 13,
    fontWeight: '600',
  },
  summarySection: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 32,
  },
  goalsSection: {
    marginBottom: 32,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  sectionTitle: {
    color: Colors.text,
    fontSize: 20,
    fontWeight: 'bold',
  },
  seeAllText: {
    color: Colors.primary,
    fontSize: 14,
    fontWeight: '600',
  },
  goalsScroll: {
    marginHorizontal: -24,
    paddingHorizontal: 24,
  },
  emptyGoals: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 32,
    backgroundColor: Colors.card,
    borderRadius: 24,
    borderWidth: 1,
    borderColor: Colors.border,
    width: width - 48,
  },
  emptyTransactions: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 40,
    backgroundColor: Colors.card,
    borderRadius: 24,
    borderWidth: 1,
    borderColor: Colors.border,
    borderStyle: 'dashed',
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
  transferSection: {
    marginBottom: 32,
  },
  transactionsList: {
    gap: 16,
  },
  transactionItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    backgroundColor: Colors.card,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  transactionLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
  },
  transactionIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
  },
  transactionCategory: {
    color: Colors.text,
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 4,
  },
  transactionDate: {
    color: Colors.textDim,
    fontSize: 13,
  },
  transactionAmount: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  quickActionsSection: {
    marginBottom: 32,
  },
  quickActionsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 16,
  },
  quickActionButton: {
    alignItems: 'center',
    width: '30%',
  },
  quickActionGradient: {
    width: 64,
    height: 64,
    borderRadius: 32,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.05)',
  },
  quickActionText: {
    color: Colors.text,
    fontSize: 13,
    fontWeight: '500',
    textAlign: 'center',
  },
  fab: {
    position: 'absolute',
    bottom: 100,
    right: 24,
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: Colors.primary,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: Colors.primary,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.4,
    shadowRadius: 12,
    elevation: 8,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.2)',
  },
});
