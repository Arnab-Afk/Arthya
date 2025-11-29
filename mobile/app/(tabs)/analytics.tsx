import React, { useState, useCallback } from 'react';
import { StyleSheet, View, Text, ScrollView, TouchableOpacity, ActivityIndicator, RefreshControl } from 'react-native';
import { Colors } from '@/constants/Colors';
import { Ionicons } from '@expo/vector-icons';
import { Svg, Circle, G } from 'react-native-svg';
import Animated, { FadeInDown, ZoomIn } from 'react-native-reanimated';
import { useRouter } from 'expo-router';
import { useFocusEffect } from '@react-navigation/native';
import api from '@/services/api';
import { CategorySpending, Transaction } from '@/types/api';

interface AnalyticsData {
    totalIncome: number;
    totalExpense: number;
    categoryBreakdown: CategorySpending[];
}

export default function AnalyticsScreen() {
    const router = useRouter();
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
    const [transactions, setTransactions] = useState<Transaction[]>([]);

    // Refresh data when screen comes into focus
    useFocusEffect(
        useCallback(() => {
            loadData();
        }, [])
    );

    const loadData = async () => {
        try {
            const [analyticsRes, transactionsRes] = await Promise.all([
                api.getSpendingAnalysis(30),
                api.getTransactions({ limit: 5 })
            ]);
            
            console.log('Analytics response:', JSON.stringify(analyticsRes, null, 2));
            console.log('Transactions response:', JSON.stringify(transactionsRes, null, 2));
            
            if (analyticsRes.success && analyticsRes.data) {
                const data = analyticsRes.data;
                setAnalytics({
                    totalIncome: data.totalIncome || 0,
                    totalExpense: data.totalSpending || data.totalExpense || 0,
                    categoryBreakdown: data.categories || data.categoryBreakdown || [],
                });
            }
            if (transactionsRes.success && transactionsRes.data) {
                // Handle both array and object with transactions property
                const txns = Array.isArray(transactionsRes.data) 
                    ? transactionsRes.data 
                    : transactionsRes.data.transactions || [];
                setTransactions(txns);
            }
        } catch (error) {
            console.error('Failed to load analytics:', error);
        } finally {
            setLoading(false);
        }
    };

    const onRefresh = async () => {
        setRefreshing(true);
        await loadData();
        setRefreshing(false);
    };

    const formatCurrency = (amount: number) => {
        return `₹${Math.abs(amount).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    };

    const getCategoryIcon = (category: string): keyof typeof Ionicons.glyphMap => {
        const icons: Record<string, keyof typeof Ionicons.glyphMap> = {
            'food': 'restaurant-outline',
            'food_dining': 'restaurant-outline',
            'shopping': 'cart-outline',
            'transport': 'car-outline',
            'transportation': 'car-outline',
            'entertainment': 'game-controller-outline',
            'utilities': 'flash-outline',
            'healthcare': 'medkit-outline',
            'health': 'medkit-outline',
            'income': 'wallet-outline',
            'other': 'ellipsis-horizontal-outline',
        };
        return icons[category.toLowerCase()] || 'ellipsis-horizontal-outline';
    };

    const getCategoryColor = (index: number): string => {
        const colors = ['#9FE8AE', '#D4B483', '#64D2FF', '#FF9500', '#FF453A', '#BF5AF2'];
        return colors[index % colors.length];
    };

    if (loading) {
        return (
            <View style={[styles.container, styles.centerContent]}>
                <ActivityIndicator size="large" color={Colors.primary} />
                <Text style={styles.loadingText}>Loading analytics...</Text>
            </View>
        );
    }

    const topCategories = analytics?.categoryBreakdown?.slice(0, 3) || [];
    const totalSpending = analytics?.totalExpense || 0;
    const savingsRate = totalSpending > 0 && analytics?.totalIncome 
        ? Math.round(((analytics.totalIncome - totalSpending) / analytics.totalIncome) * 100)
        : 0;

    return (
        <View style={styles.container}>
            <Animated.View entering={FadeInDown.delay(100).duration(500)} style={styles.header}>
                <View style={{ width: 24 }} />
                <Text style={styles.headerTitle}>Analytics</Text>
                <TouchableOpacity onPress={() => router.push('/explore')}>
                    <Ionicons name="person-circle-outline" size={28} color={Colors.text} />
                </TouchableOpacity>
            </Animated.View>

            <ScrollView 
                contentContainerStyle={styles.content}
                refreshControl={
                    <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={Colors.primary} />
                }
            >
                {/* Donut Chart */}
                <Animated.View entering={ZoomIn.delay(300).duration(600)} style={styles.chartContainer}>
                    <View style={{ position: 'relative', width: 280, height: 280, alignItems: 'center', justifyContent: 'center' }}>
                        <Svg height="280" width="280" viewBox="0 0 280 280" style={{ position: 'absolute' }}>
                            <G rotation="-90" origin="140, 140">
                                {/* Background Circle */}
                                <Circle
                                    cx="140"
                                    cy="140"
                                    r="90"
                                    stroke="#1C1C1E"
                                    strokeWidth="50"
                                    fill="transparent"
                                />
                                {/* Progress Circle - Savings Rate */}
                                <Circle
                                    cx="140"
                                    cy="140"
                                    r="90"
                                    stroke={Colors.primary}
                                    strokeWidth="50"
                                    fill="transparent"
                                    strokeDasharray="565"
                                    strokeDashoffset={565 * (1 - Math.max(0, savingsRate) / 100)}
                                    strokeLinecap="butt"
                                />
                            </G>
                        </Svg>
                        {/* Center Text */}
                        <View style={{ alignItems: 'center', justifyContent: 'center' }}>
                            <Text style={styles.percentageText}>{savingsRate}%</Text>
                            <Text style={styles.balanceText}>Savings Rate</Text>
                        </View>
                    </View>
                </Animated.View>

                {/* Summary Cards */}
                <Animated.View entering={FadeInDown.delay(400).duration(500)} style={styles.summaryRow}>
                    <View style={styles.summaryCard}>
                        <Text style={styles.summaryLabel}>Income</Text>
                        <Text style={[styles.summaryValue, { color: '#9FE8AE' }]}>
                            {formatCurrency(analytics?.totalIncome || 0)}
                        </Text>
                    </View>
                    <View style={styles.summaryCard}>
                        <Text style={styles.summaryLabel}>Expenses</Text>
                        <Text style={[styles.summaryValue, { color: '#FF453A' }]}>
                            {formatCurrency(analytics?.totalExpense || 0)}
                        </Text>
                    </View>
                </Animated.View>

                {/* Top Categories */}
                <Animated.View entering={FadeInDown.delay(500).duration(500)} style={styles.statsRow}>
                    {topCategories.length > 0 ? (
                        topCategories.map((cat, index) => (
                            <View key={cat.category} style={styles.statCard}>
                                <Ionicons name={getCategoryIcon(cat.category)} size={24} color={getCategoryColor(index)} />
                                <Text style={styles.statLabel}>{cat.category}</Text>
                                <Text style={styles.statValue}>{parseFloat(cat.percentage)?.toFixed(0) || 0}%</Text>
                            </View>
                        ))
                    ) : (
                        <View style={styles.emptyState}>
                            <Text style={styles.emptyText}>No spending data yet</Text>
                        </View>
                    )}
                </Animated.View>

                {/* Recent Transactions */}
                <Animated.View entering={FadeInDown.delay(600).duration(500)}>
                    <Text style={styles.sectionTitle}>Recent Transactions</Text>
                    {transactions.length > 0 ? (
                        transactions.map((transaction) => (
                            <View key={transaction.id || transaction._id} style={styles.transactionItem}>
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
                                <View style={styles.transactionDetails}>
                                    <Text style={styles.transactionName}>{transaction.category}</Text>
                                    <Text style={styles.transactionDate}>
                                        {new Date(transaction.date).toLocaleDateString('en-US', {
                                            day: 'numeric',
                                            month: 'short',
                                            year: 'numeric'
                                        })}
                                    </Text>
                                </View>
                                <Text style={[
                                    styles.transactionAmount,
                                    { color: transaction.type === 'income' ? '#9FE8AE' : '#FF453A' }
                                ]}>
                                    {transaction.type === 'income' ? '+' : '-'} {formatCurrency(transaction.amount)}
                                </Text>
                            </View>
                        ))
                    ) : (
                        <View style={styles.emptyState}>
                            <Ionicons name="receipt-outline" size={48} color={Colors.textDim} />
                            <Text style={styles.emptyText}>No transactions yet</Text>
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
                <Ionicons name="add" size={28} color="#000" />
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
    headerTitle: {
        fontSize: 20,
        fontWeight: '600',
        color: Colors.text,
    },
    content: {
        paddingHorizontal: 24,
        paddingBottom: 100,
    },
    chartContainer: {
        alignItems: 'center',
        marginBottom: 24,
        marginTop: 20,
    },
    percentageText: {
        fontSize: 48,
        fontWeight: 'bold',
        color: Colors.text,
    },
    balanceText: {
        color: Colors.textDim,
        textAlign: 'center',
        width: 120,
        fontSize: 14,
        marginTop: 8,
    },
    summaryRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: 24,
        gap: 16,
    },
    summaryCard: {
        flex: 1,
        backgroundColor: Colors.card,
        padding: 16,
        borderRadius: 16,
        alignItems: 'center',
        borderWidth: 1,
        borderColor: Colors.border,
    },
    summaryLabel: {
        color: Colors.textDim,
        fontSize: 14,
        marginBottom: 8,
    },
    summaryValue: {
        fontSize: 20,
        fontWeight: 'bold',
    },
    statsRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: 32,
    },
    statCard: {
        backgroundColor: Colors.card,
        padding: 16,
        borderRadius: 24,
        alignItems: 'center',
        width: '30%',
        height: 100,
        justifyContent: 'center',
        borderWidth: 1,
        borderColor: Colors.border,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 3 },
        shadowOpacity: 0.12,
        shadowRadius: 5,
        elevation: 3,
    },
    statLabel: {
        color: Colors.textDim,
        marginTop: 8,
        fontSize: 11,
        textTransform: 'capitalize',
    },
    statValue: {
        color: Colors.text,
        fontWeight: 'bold',
        marginTop: 4,
        fontSize: 16,
    },
    sectionTitle: {
        fontSize: 20,
        fontWeight: '600',
        color: Colors.text,
        marginBottom: 20,
    },
    transactionItem: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: 16,
        backgroundColor: Colors.card,
        padding: 16,
        borderRadius: 16,
        borderWidth: 1,
        borderColor: Colors.border,
    },
    transactionIcon: {
        width: 48,
        height: 48,
        borderRadius: 24,
        justifyContent: 'center',
        alignItems: 'center',
        marginRight: 16,
    },
    transactionDetails: {
        flex: 1,
    },
    transactionName: {
        color: Colors.text,
        fontWeight: '500',
        fontSize: 16,
        textTransform: 'capitalize',
    },
    transactionDate: {
        color: Colors.textDim,
        fontSize: 12,
        marginTop: 4,
    },
    transactionAmount: {
        fontWeight: '600',
        fontSize: 16,
    },
    emptyState: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        padding: 40,
    },
    emptyText: {
        color: Colors.textDim,
        fontSize: 14,
        marginTop: 12,
    },
    fab: {
        position: 'absolute',
        bottom: 100,
        right: 24,
        width: 60,
        height: 60,
        borderRadius: 30,
        backgroundColor: Colors.primary,
        justifyContent: 'center',
        alignItems: 'center',
        shadowColor: Colors.primary,
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.4,
        shadowRadius: 8,
        elevation: 8,
    },
});
